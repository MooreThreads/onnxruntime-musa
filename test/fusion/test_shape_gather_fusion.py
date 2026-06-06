# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for shape metadata without Shape-Gather fusion copy nodes."""

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


def _build_shape_metadata_reshape_model() -> bytes:
    gather_index = numpy_helper.from_array(
        np.array([0, 1], dtype=np.int64), name="gather_index"
    )
    suffix = numpy_helper.from_array(np.array([4], dtype=np.int32), name="suffix")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node("Cast", ["shape_i64"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Gather", ["shape_i32", "gather_index"], ["prefix"], axis=0),
        helper.make_node("Concat", ["prefix", "suffix"], ["target_i32"], axis=0),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["Data", "target_i64"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_metadata_reshape",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M", "K"]),
            helper.make_tensor_value_info("Data", TensorProto.FLOAT, ["N", "MK"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[gather_index, suffix],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


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


def test_shape_metadata_reshape_has_no_memcpy_to_host(tmp_path):
    model = _build_shape_metadata_reshape_model()
    x = np.zeros((2, 3, 4), dtype=np.float32)
    data = np.arange(24, dtype=np.float32).reshape(2, 12)

    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, {"X": x, "Data": data})

    (actual,), events = _profile_musa_session(
        model, {"X": x, "Data": data}, tmp_path, "shape_metadata_reshape"
    )
    np.testing.assert_array_equal(actual, expected)

    ops_by_provider = _ops_by_provider(events)
    musa_ops = ops_by_provider.get("MUSAExecutionProvider", set())
    cpu_ops = ops_by_provider.get("CPUExecutionProvider", set())
    all_ops = set().union(*ops_by_provider.values())

    assert {"Shape", "Reshape"} <= musa_ops
    assert {"Cast", "Gather", "Concat"} <= cpu_ops
    assert "Gather" not in musa_ops
    assert "Concat" not in musa_ops
    assert "MemcpyToHost" not in all_ops
    assert "MemcpyFromHost" not in all_ops
    assert not any(str(op).startswith("MUSAExecutionProvider_") for op in all_ops)


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
