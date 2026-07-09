# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end test for the SparseIdToMask fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_sparse_id_to_mask_model() -> bytes:
    default_id = numpy_helper.from_array(
        np.array(-1, dtype=np.int64), name="default_id"
    )
    nodes = [
        helper.make_node(
            "LessOrEqual", ["bound", "SparseIds"], ["Cond"], name="LessEqual"
        ),
        helper.make_node("Not", ["Cond"], ["NotCond"], name="SelectV2__not"),
        helper.make_node(
            "Cast",
            ["NotCond"],
            ["NotCondI64"],
            name="SelectV2__not_cast",
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Mul", ["bound", "NotCondI64"], ["BoundPart"], name="SelectV2__bound"
        ),
        helper.make_node(
            "Cast",
            ["Cond"],
            ["CondI64"],
            name="SelectV2__cond_cast",
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Mul",
            ["CondI64", "default_id"],
            ["DefaultPart"],
            name="SelectV2__default",
        ),
        helper.make_node(
            "Add", ["DefaultPart", "BoundPart"], ["Candidate"], name="SelectV2"
        ),
        helper.make_node("Equal", ["DenseIds", "Candidate"], ["MaskBool"], name="Equal"),
        helper.make_node(
            "Cast", ["MaskBool"], ["Mask"], name="CastMask", to=TensorProto.INT32
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "sparse_id_to_mask_fusion",
        [
            helper.make_tensor_value_info("DenseIds", TensorProto.INT64, ["N", 4]),
            helper.make_tensor_value_info("bound", TensorProto.INT64, ["N", 1]),
            helper.make_tensor_value_info("SparseIds", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("Mask", TensorProto.INT32, None)],
        initializer=[default_id],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_sparse_id_to_mask_fusion(tmp_path):
    model = _build_sparse_id_to_mask_model()
    dense_ids = np.array([[5, -1, 5, -1], [5, -1, -1, 5]], dtype=np.int64)
    bound = np.array([[5], [4]], dtype=np.int64)
    sparse_ids = np.array(7, dtype=np.int64)
    feeds = {"DenseIds": dense_ids, "bound": bound, "SparseIds": sparse_ids}

    (actual,) = run_model_and_compare(model, feeds)
    candidate = np.where(bound <= sparse_ids, np.int64(-1), bound)
    expected = (dense_ids == candidate).astype(np.int32)
    np.testing.assert_allclose(actual, expected)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "sparse_id_to_mask_fusion")
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
    assert not ({"LessOrEqual", "Not", "Cast", "Mul", "Add", "Equal"} & op_names)
