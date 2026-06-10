# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_centered_reduce_model() -> bytes:
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")
    nodes = [
        helper.make_node("ReduceSum", ["X", "axes"], ["R0"], keepdims=1),
        helper.make_node("Sub", ["X", "R0"], ["C"]),
        helper.make_node("Mul", ["C", "C"], ["S"]),
        helper.make_node("ReduceSum", ["S", "axes"], ["R1"], keepdims=1),
    ]
    graph = helper.make_graph(
        nodes,
        "centered_reduce_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3, 4])],
        [
            helper.make_tensor_value_info("R0", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("R1", TensorProto.FLOAT, None),
        ],
        initializer=[axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_centered_reduce_fusion(tmp_path):
    model = _build_centered_reduce_model()
    x = np.linspace(-1.0, 1.0, num=2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    feeds = {"X": x}

    actual_r0, actual_r1 = run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    expected_r0 = np.sum(x, axis=2, keepdims=True)
    centered = x - expected_r0
    expected_r1 = np.sum(centered * centered, axis=2, keepdims=True)
    np.testing.assert_allclose(actual_r0, expected_r0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_r1, expected_r1, rtol=1e-5, atol=1e-5)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "centered_reduce_fusion")
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
    assert not ({"ReduceSum", "Sub", "Mul"} & op_names)
