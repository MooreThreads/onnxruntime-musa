# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for Pow + Add/Sub feeding SplitReduce."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_model(affine_op: str) -> bytes:
    split = numpy_helper.from_array(np.array([3, 5], dtype=np.int64), name="split")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    nodes = [
        helper.make_node("Pow", ["X", "E"], ["P"]),
        helper.make_node(affine_op, ["P", "B"], ["A"]),
        helper.make_node("Split", ["A", "split"], ["A0", "A1"], axis=1),
        helper.make_node("ReduceProd", ["A0", "axes"], ["Y0"], keepdims=0),
        helper.make_node("ReduceMean", ["A1", "axes"], ["Y1"], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        f"pow_{affine_op.lower()}_split_reduce_fusion",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 8, 4]),
            helper.make_tensor_value_info("E", TensorProto.FLOAT, ["N", 8, 1]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, ["N", 8, 1]),
        ],
        [
            helper.make_tensor_value_info("Y0", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("Y1", TensorProto.FLOAT, None),
        ],
        initializer=[split, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


@pytest.mark.parametrize("affine_op", ["Add", "Sub"])
def test_pow_affine_split_reduce_fusion(affine_op, tmp_path):
    model = _build_model(affine_op)
    x = np.linspace(0.8, 1.3, num=2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
    exponent = np.linspace(0.9, 1.1, num=2 * 8, dtype=np.float32).reshape(2, 8, 1)
    bias = np.linspace(0.01, 0.02, num=2 * 8, dtype=np.float32).reshape(2, 8, 1)
    feeds = {"X": x, "E": exponent, "B": bias}

    actual_prod, actual_mean = run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    affine = np.power(x, exponent)
    if affine_op == "Add":
        affine = affine + bias
    else:
        affine = affine - bias
    np.testing.assert_allclose(actual_prod, np.prod(affine[:, :3, :], axis=1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_mean, np.mean(affine[:, 3:, :], axis=1), rtol=1e-4, atol=1e-4)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "pow_affine_split_reduce_fusion")
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
    assert not ({"Pow", affine_op, "Split", "ReduceProd", "ReduceMean"} & op_names)
