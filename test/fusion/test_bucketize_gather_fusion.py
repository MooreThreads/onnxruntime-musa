# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end test for the BucketizeGather fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_bucketize_gather_model() -> bytes:
    modulus = numpy_helper.from_array(np.array(10, dtype=np.int64), name="modulus")
    offset = numpy_helper.from_array(np.array(1, dtype=np.int64), name="offset")
    threshold = numpy_helper.from_array(
        np.array([0], dtype=np.float32), name="threshold"
    )
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    table = numpy_helper.from_array(
        np.arange(11 * 6, dtype=np.float32).reshape(11, 6), name="table"
    )
    nodes = [
        helper.make_node("Cast", ["Ids"], ["IdsFloat"], to=TensorProto.FLOAT),
        helper.make_node("Greater", ["IdsFloat", "threshold"], ["IsPositive"]),
        helper.make_node(
            "Cast", ["IsPositive"], ["Mask"], to=TensorProto.INT64, name="Cast_7"
        ),
        helper.make_node("Div", ["Ids", "modulus"], ["DivOut"], name="Div__10131"),
        helper.make_node(
            "Mul", ["DivOut", "modulus"], ["MulOut"], name="Mul__10133"
        ),
        helper.make_node("Sub", ["Ids", "MulOut"], ["Remainder"], name="FloorMod_7"),
        helper.make_node("Add", ["Remainder", "offset"], ["Shifted"], name="add_7"),
        helper.make_node("Mul", ["Shifted", "Mask"], ["GatherIds"], name="mul_28"),
        helper.make_node(
            "Gather",
            ["table", "GatherIds"],
            ["GatherOut"],
            axis=0,
            name="embedding_lookup_7",
        ),
        helper.make_node(
            "Squeeze", ["GatherOut", "axes"], ["Y"], name="Squeeze_7"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "bucketize_gather_fusion",
        [helper.make_tensor_value_info("Ids", TensorProto.INT64, ["N", 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[modulus, offset, threshold, axes, table],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_bucketize_gather_fusion(tmp_path):
    model = _build_bucketize_gather_model()
    ids = np.array([[0], [1], [10], [11], [-1], [23]], dtype=np.int64)
    feeds = {"Ids": ids}

    (actual,) = run_model_and_compare(model, feeds, rtol=0, atol=0)
    table = np.arange(11 * 6, dtype=np.float32).reshape(11, 6)
    gather_ids = np.where(ids.astype(np.float32) > 0, (ids - (ids // 10) * 10) + 1, 0)
    expected = np.squeeze(table[gather_ids], axis=1)
    np.testing.assert_array_equal(actual, expected)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "bucketize_gather_fusion")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)

    node_events = [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    ]
    op_names = {e.get("args", {}).get("op_name") for e in node_events}
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not (
        {"Cast", "Greater", "Div", "Mul", "Sub", "Add", "Gather", "Squeeze"}
        & op_names
    )
