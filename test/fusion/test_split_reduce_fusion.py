# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for Split(axis=1) feeding ReduceProd/ReduceMean."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_split_reduce_model() -> bytes:
    split = numpy_helper.from_array(np.array([10, 20], dtype=np.int64), name="split")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    nodes = [
        helper.make_node("Split", ["X", "split"], ["X0", "X1"], axis=1),
        helper.make_node("ReduceProd", ["X0", "axes"], ["Y0"], keepdims=0),
        helper.make_node("ReduceMean", ["X1", "axes"], ["Y1"], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "split_reduce_fusion_profile",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 30, 4])],
        [
            helper.make_tensor_value_info("Y0", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("Y1", TensorProto.FLOAT, None),
        ],
        initializer=[split, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_split_reduce_fusion_removes_split_reduce_chain(tmp_path):
    model = _build_split_reduce_model()
    feeds = {
        "X": np.linspace(0.5, 2.0, num=2 * 30 * 4, dtype=np.float32).reshape(
            2, 30, 4
        )
    }

    actual_prod, actual_mean = run_model_and_compare(model, feeds)
    expected_prod = np.prod(feeds["X"][:, :10, :], axis=1)
    expected_mean = np.mean(feeds["X"][:, 10:, :], axis=1)
    np.testing.assert_allclose(actual_prod, expected_prod, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-5, atol=1e-5)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "split_reduce_fusion")
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
    providers = {e.get("args", {}).get("provider") for e in node_events}
    assert "MUSAExecutionProvider" in providers
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"Split", "ReduceProd", "ReduceMean"} & op_names)
