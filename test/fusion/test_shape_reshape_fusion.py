# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for Shape/Cast/Split/Concat/Cast feeding Reshape."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_shape_reshape_model() -> bytes:
    tail = numpy_helper.from_array(np.array([-1], dtype=np.int32), name="tail")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Split", ["shape_i32"], ["dim0", "dim1", "dim2"], axis=0),
        helper.make_node("Concat", ["dim0", "dim1", "tail"], ["target_i32"], axis=0),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["X", "target_i64"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_reshape_fusion_profile",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M", "K"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[tail],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_shape_tile_reshape_model() -> bytes:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    tile_tail = numpy_helper.from_array(np.array([2], dtype=np.int32), name="tile_tail")
    reshape_tail = numpy_helper.from_array(
        np.array([4], dtype=np.int32), name="reshape_tail"
    )
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Slice", ["shape_i32", "starts", "ends", "axes"], ["dim0"]),
        helper.make_node("Concat", ["dim0", "tile_tail"], ["repeats_i32"], axis=0),
        helper.make_node("Cast", ["repeats_i32"], ["repeats"], to=TensorProto.INT64),
        helper.make_node("Tile", ["D", "repeats"], ["T"]),
        helper.make_node("Concat", ["dim0", "reshape_tail"], ["target_i32"], axis=0),
        helper.make_node("Cast", ["target_i32"], ["target"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["T", "target"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_tile_reshape_fusion_profile",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M"]),
            helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 2]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
        ],
        initializer=[starts, ends, axes, tile_tail, reshape_tail],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_shape_reshape_fusion_removes_metadata_chain(tmp_path):
    model = _build_shape_reshape_model()
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    (actual,) = run_model_and_compare(model, {"X": x})
    np.testing.assert_array_equal(actual, x)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_reshape_fusion")
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
    assert not ({"Shape", "Cast", "Split", "Concat", "Reshape"} & op_names)


def test_shape_reshape_fusion_handles_tile_repeats(tmp_path):
    model = _build_shape_tile_reshape_model()
    feeds = {
        "X": np.ones((3, 5), dtype=np.float32),
        "D": np.array([[1.0, 2.0]], dtype=np.float32),
    }

    (actual,) = run_model_and_compare(model, feeds)
    np.testing.assert_array_equal(actual, np.tile(feeds["D"], (3, 2)).reshape(3, 4))

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_tile_reshape_fusion")
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
    assert not ({"Shape", "Cast", "Slice", "Concat", "Tile", "Reshape"} & op_names)
