# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Tests for shape Cast + Reshape + Split fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_shape_cast_split_model() -> bytes:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    infer = numpy_helper.from_array(np.array([-1], dtype=np.int32), name="infer")
    width = numpy_helper.from_array(np.array([768], dtype=np.int32), name="width")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Slice", ["shape_i32", "starts", "ends", "axes"], ["dim0"]),
        helper.make_node("Concat", ["dim0", "infer", "width"], ["target_i32"], axis=0),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["X", "target_i64"], ["R"]),
        helper.make_node("Split", ["R"], ["Y0", "Y1", "Y2"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_cast_split_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3, 768])],
        [
            helper.make_tensor_value_info("Y0", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("Y1", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("Y2", TensorProto.FLOAT, None),
        ],
        initializer=[starts, ends, axes, infer, width],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_shape_cast_split_fusion_removes_shape_d2h_and_intermediate_ops(tmp_path):
    model = _build_shape_cast_split_model()
    x = np.arange(2 * 3 * 768, dtype=np.float32).reshape(2, 3, 768)

    actual_outputs = run_model_and_compare(model, {"X": x})
    expected_outputs = np.split(x.reshape(2, 3, 768), 3, axis=1)
    for actual, expected in zip(actual_outputs, expected_outputs):
        np.testing.assert_array_equal(actual, expected)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_cast_split_fusion")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    session.run(None, {"X": x})
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
    assert "MemcpyToHost" not in op_names
    assert "Reshape" not in op_names
    assert "Split" not in op_names
