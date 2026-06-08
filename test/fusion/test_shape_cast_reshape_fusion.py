# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for terminal Concat/Cast shape tensors feeding Reshape."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_terminal_shape_cast_reshape_model() -> bytes:
    tail = numpy_helper.from_array(np.array([3], dtype=np.int32), name="tail")
    scale = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="scale")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Split", ["shape_i32"], ["dim0", "dim1"], axis=0),
        helper.make_node("Concat", ["dim0", "tail"], ["target1_i32"], axis=0),
        helper.make_node(
            "Cast", ["target1_i32"], ["target1_i64"], to=TensorProto.INT64
        ),
        helper.make_node("Reshape", ["X", "target1_i64"], ["Y1"]),
        helper.make_node("Mul", ["Y1", "scale"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "terminal_shape_cast_reshape_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[tail, scale],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_terminal_shape_cast_reshape_with_infer_model() -> bytes:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    infer = numpy_helper.from_array(np.array([-1], dtype=np.int32), name="infer")
    head_count = numpy_helper.from_array(
        np.array([2], dtype=np.int32), name="head_count"
    )
    head_size = numpy_helper.from_array(
        np.array([2], dtype=np.int32), name="head_size"
    )
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Slice", ["shape_i32", "starts", "ends", "axes"], ["dim0"]),
        helper.make_node(
            "Concat",
            ["dim0", "infer", "head_count", "head_size"],
            ["target_i32"],
            axis=0,
        ),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["X", "target_i64"], ["Y1"]),
        helper.make_node("Reshape", ["X", "target_i64"], ["Y2"]),
        helper.make_node("Add", ["Y1", "Y2"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "terminal_shape_cast_reshape_infer_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[starts, ends, axes, infer, head_count, head_size],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_shape_cast_reshape_external_shape_source_model() -> bytes:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    infer = numpy_helper.from_array(np.array([-1], dtype=np.int32), name="infer")
    head_count = numpy_helper.from_array(
        np.array([3], dtype=np.int32), name="head_count"
    )
    head_size = numpy_helper.from_array(
        np.array([2], dtype=np.int32), name="head_size"
    )
    scale = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="scale")
    nodes = [
        helper.make_node("Shape", ["S"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Slice", ["shape_i32", "starts", "ends", "axes"], ["dim0"]),
        helper.make_node(
            "Concat",
            ["dim0", "infer", "head_count", "head_size"],
            ["target_i32"],
            axis=0,
        ),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["X", "target_i64"], ["Y1"]),
        helper.make_node("Mul", ["Y1", "scale"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_cast_reshape_external_shape_source",
        [
            helper.make_tensor_value_info("S", TensorProto.FLOAT, ["N", 4, 6]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["M", 6]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[starts, ends, axes, infer, head_count, head_size, scale],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_terminal_shape_cast_reshape_fusion_removes_d2h_shape_copy(tmp_path):
    model = _build_terminal_shape_cast_reshape_model()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    (actual,) = run_model_and_compare(model, {"X": x})
    np.testing.assert_array_equal(actual, x * 2.0)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_cast_reshape_fusion")
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
    providers = {e.get("args", {}).get("provider") for e in node_events}
    assert "MUSAExecutionProvider" in providers
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert "MemcpyToHost" not in op_names
    assert "Reshape" not in op_names

def test_shape_cast_reshape_uses_shape_source_dim_for_external_data(tmp_path):
    model = _build_shape_cast_reshape_external_shape_source_model()
    feeds = {
        "S": np.zeros((2, 4, 6), dtype=np.float32),
        "X": np.arange(8 * 6, dtype=np.float32).reshape(8, 6),
    }

    (actual,) = run_model_and_compare(model, feeds)
    np.testing.assert_array_equal(actual, feeds["X"].reshape(2, 4, 3, 2))

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_cast_reshape_external_source")
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
    assert "Reshape" not in op_names


def test_terminal_shape_cast_reshape_fusion_allows_dynamic_batch_with_infer(
    tmp_path,
):
    model = _build_terminal_shape_cast_reshape_with_infer_model()
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    (actual,) = run_model_and_compare(model, {"X": x})
    np.testing.assert_array_equal(actual, x.reshape(2, 3, 2, 2) * 2.0)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_cast_reshape_infer_fusion")
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
