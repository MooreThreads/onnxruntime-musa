# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Tests for metadata-only Shape/Cast/Slice/Concat/Cast fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_shared_shape_concat_model() -> bytes:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    qkv_tail = numpy_helper.from_array(
        np.array([-1, 12, 64], dtype=np.int32), name="qkv_tail"
    )
    merge_tail = numpy_helper.from_array(
        np.array([-1, 768], dtype=np.int32), name="merge_tail"
    )

    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node("Cast", ["shape_i64"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node(
            "Slice", ["shape_i32", "starts", "ends", "axes"], ["batch_i32"]
        ),
        helper.make_node("Concat", ["batch_i32", "qkv_tail"], ["qkv_shape_i32"], axis=0),
        helper.make_node(
            "Cast", ["qkv_shape_i32"], ["qkv_shape_i64"], to=TensorProto.INT64
        ),
        helper.make_node("Reshape", ["X", "qkv_shape_i64"], ["Y1"]),
        helper.make_node(
            "Concat", ["batch_i32", "merge_tail"], ["merge_shape_i32"], axis=0
        ),
        helper.make_node(
            "Cast",
            ["merge_shape_i32"],
            ["merge_shape_i64"],
            to=TensorProto.INT64,
        ),
        helper.make_node("Reshape", ["X", "merge_shape_i64"], ["Y2"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_cast_concat_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 13, 768])],
        [
            helper.make_tensor_value_info("Y1", TensorProto.FLOAT, ["N", 13, 12, 64]),
            helper.make_tensor_value_info("Y2", TensorProto.FLOAT, ["N", 13, 768]),
        ],
        initializer=[starts, ends, axes, qkv_tail, merge_tail],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_shape_cast_concat_fusion_shared_slice(tmp_path):
    model = _build_shared_shape_concat_model()
    x = np.arange(2 * 13 * 768, dtype=np.float32).reshape(2, 13, 768)

    actual_y1, actual_y2 = run_model_and_compare(model, {"X": x})
    np.testing.assert_array_equal(actual_y1, x.reshape(2, 13, 12, 64))
    np.testing.assert_array_equal(actual_y2, x)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_cast_concat_fusion")
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
