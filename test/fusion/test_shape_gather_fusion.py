# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Guardrails for shape-metadata matching not catching large data tensors."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices


def _profile_musa_session(model: bytes, feeds: dict[str, np.ndarray], tmp_path, prefix: str):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    outputs = session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    return outputs, events


def _node_events(events):
    return [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    ]


def _ops_by_provider(events):
    ops = {}
    for event in _node_events(events):
        args = event.get("args", {})
        provider = args.get("provider")
        op_name = args.get("op_name")
        ops.setdefault(provider, set()).add(op_name)
    return ops


def _build_large_cast_model() -> bytes:
    node = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT)
    graph = helper.make_graph(
        [node],
        "large_cast",
        [helper.make_tensor_value_info("X", TensorProto.INT32, [1024, 60, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1024, 60, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_constant_of_shape_gather_model() -> bytes:
    embedding = numpy_helper.from_array(
        np.arange(8, dtype=np.float32).reshape(2, 4), name="embedding"
    )
    zero = numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("ConstantOfShape", ["shape"], ["indices"], value=zero),
        helper.make_node("Gather", ["embedding", "indices"], ["Y"], axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        "constant_of_shape_gather",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[embedding],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_shape_gather_expand_model() -> bytes:
    scalar_index = numpy_helper.from_array(
        np.array(0, dtype=np.int64), name="scalar_index"
    )
    vector_index = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="vector_index"
    )
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Gather", ["shape", "scalar_index"], ["batch_scalar"], axis=0),
        helper.make_node("Gather", ["shape", "vector_index"], ["target_shape"], axis=0),
        helper.make_node("Expand", ["batch_scalar", "target_shape"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_gather_expand",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M"])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, None)],
        initializer=[scalar_index, vector_index],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_shape_mul_expand_shape_model() -> bytes:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    two = numpy_helper.from_array(np.array(2, dtype=np.int32), name="two")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node("Cast", ["shape_i64"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Slice", ["shape_i32", "starts", "ends", "axes"], ["batch_vec"]),
        helper.make_node("Squeeze", ["batch_vec", "axes"], ["batch_scalar"]),
        helper.make_node("Mul", ["batch_scalar", "two"], ["double_batch"]),
        helper.make_node("Unsqueeze", ["double_batch", "axes"], ["target_i32"]),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], to=TensorProto.INT64),
        helper.make_node("Expand", ["Data", "target_i64"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_mul_expand_shape",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M"]),
            helper.make_tensor_value_info("Data", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[starts, ends, axes, two],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_large_cast_stays_on_musa(tmp_path):
    model = _build_large_cast_model()
    x = np.arange(1024 * 60, dtype=np.int32).reshape(1024, 60, 1)

    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, {"X": x})

    (actual,), events = _profile_musa_session(model, {"X": x}, tmp_path, "large_cast")
    np.testing.assert_array_equal(actual, expected)

    ops_by_provider = _ops_by_provider(events)
    assert "Cast" in ops_by_provider.get("MUSAExecutionProvider", set())
    assert "Cast" not in ops_by_provider.get("CPUExecutionProvider", set())


def test_constant_of_shape_gather_indices_stay_on_musa(tmp_path):
    model = _build_constant_of_shape_gather_model()
    x = np.zeros((2, 3), dtype=np.float32)

    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, {"X": x})

    (actual,), events = _profile_musa_session(
        model, {"X": x}, tmp_path, "constant_of_shape_gather"
    )
    np.testing.assert_array_equal(actual, expected)

    ops_by_provider = _ops_by_provider(events)
    musa_ops = ops_by_provider.get("MUSAExecutionProvider", set())
    cpu_ops = ops_by_provider.get("CPUExecutionProvider", set())
    assert "Gather" in musa_ops or any(
        str(op).startswith("MUSAExecutionProvider_") for op in musa_ops
    )
    assert "ConstantOfShape" not in cpu_ops


def test_shape_gather_feeding_expand_stays_on_musa(tmp_path):
    model = _build_shape_gather_expand_model()
    x = np.zeros((3, 4), dtype=np.float32)

    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, {"X": x})

    (actual,), events = _profile_musa_session(
        model, {"X": x}, tmp_path, "shape_gather_expand"
    )
    np.testing.assert_array_equal(actual, expected)

    ops_by_provider = _ops_by_provider(events)
    musa_ops = ops_by_provider.get("MUSAExecutionProvider", set())
    cpu_ops = ops_by_provider.get("CPUExecutionProvider", set())
    assert "Gather" in musa_ops
    assert "Expand" in musa_ops
    assert "Expand" not in cpu_ops


def test_int32_shape_mul_feeding_expand_shape_uses_metadata_fusion(tmp_path):
    model = _build_shape_mul_expand_shape_model()
    feeds = {
        "X": np.zeros((3, 4), dtype=np.float32),
        "Data": np.array(1.5, dtype=np.float32),
    }

    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, feeds)

    (actual,), events = _profile_musa_session(
        model, feeds, tmp_path, "shape_mul_expand_shape"
    )
    np.testing.assert_array_equal(actual, expected)

    ops_by_provider = _ops_by_provider(events)
    musa_ops = ops_by_provider.get("MUSAExecutionProvider", set())
    cpu_ops = ops_by_provider.get("CPUExecutionProvider", set())
    assert "Mul" not in cpu_ops
    assert "Expand" in musa_ops or any(
        str(op).startswith("MUSAExecutionProvider_") for op in musa_ops
    )
