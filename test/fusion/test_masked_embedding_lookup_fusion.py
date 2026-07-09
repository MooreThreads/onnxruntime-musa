# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end test for the MaskedEmbeddingLookup fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_masked_embedding_lookup_model() -> bytes:
    sequence = 6
    embedding_dim = 4
    table = numpy_helper.from_array(
        np.arange(8 * embedding_dim, dtype=np.float32).reshape(8, embedding_dim),
        name="table",
    )
    reshape_shape = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="shape")
    threshold = numpy_helper.from_array(np.array(0, dtype=np.int64), name="threshold")
    squeeze_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="squeeze_axes"
    )
    unsqueeze_axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="unsqueeze_axes"
    )
    zero_data = numpy_helper.from_array(
        np.zeros((sequence, embedding_dim), dtype=np.float32), name="zero_data"
    )
    nodes = [
        helper.make_node("Reshape", ["Ids", "shape"], ["FlatIds"], name="Reshape"),
        helper.make_node(
            "GreaterOrEqual", ["FlatIds", "threshold"], ["Mask"], name="GreaterEqual"
        ),
        helper.make_node("NonZero", ["Mask"], ["Where"], name="Where"),
        helper.make_node("Transpose", ["Where"], ["WhereT"], name="where_transpose"),
        helper.make_node(
            "Squeeze", ["WhereT", "squeeze_axes"], ["Positions"], name="Squeeze"
        ),
        helper.make_node(
            "Gather", ["FlatIds", "Positions"], ["ValidIds"], axis=0, name="GatherV2"
        ),
        helper.make_node(
            "Gather",
            ["table", "ValidIds"],
            ["Embeddings"],
            axis=0,
            name="embedding_lookup",
        ),
        helper.make_node(
            "ScatterND",
            ["zero_data", "WhereT", "Embeddings"],
            ["DenseEmbeddings"],
            name="ScatterNd",
        ),
        helper.make_node(
            "Unsqueeze",
            ["DenseEmbeddings", "unsqueeze_axes"],
            ["Y"],
            name="ExpandDims_4",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "masked_embedding_lookup_fusion",
        [helper.make_tensor_value_info("Ids", TensorProto.INT64, [1, sequence])],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [1, sequence, embedding_dim]
            )
        ],
        initializer=[
            table,
            reshape_shape,
            threshold,
            squeeze_axes,
            unsqueeze_axes,
            zero_data,
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_masked_embedding_lookup_fusion(tmp_path):
    model = _build_masked_embedding_lookup_model()
    ids = np.array([[3, -1, 0, 7, -1, 2]], dtype=np.int64)
    feeds = {"Ids": ids}

    (actual,) = run_model_and_compare(model, feeds, rtol=0, atol=0)
    table = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    expected = np.zeros((1, 6, 4), dtype=np.float32)
    flat = ids.reshape(-1)
    for i, value in enumerate(flat):
        if value >= 0:
            expected[0, i, :] = table[value]
    np.testing.assert_array_equal(actual, expected)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "masked_embedding_lookup_fusion")
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
        {
            "Reshape",
            "GreaterOrEqual",
            "NonZero",
            "Transpose",
            "Squeeze",
            "Gather",
            "ScatterND",
            "Unsqueeze",
        }
        & op_names
    )
