# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end tests for the Shape/Cast/Gather/Concat/Reshape fusion."""

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


def _assert_shape_reshape_fused(events):
    ops_by_provider = _ops_by_provider(events)
    musa_ops = ops_by_provider.get("MUSAExecutionProvider", set())
    cpu_ops = ops_by_provider.get("CPUExecutionProvider", set())
    all_ops = set().union(*ops_by_provider.values())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert fused_ops
    assert "Shape" not in musa_ops
    assert "Reshape" not in musa_ops
    assert "Cast" not in cpu_ops
    assert "Gather" not in cpu_ops
    assert "Concat" not in cpu_ops
    assert "Gather" not in musa_ops
    assert "Concat" not in musa_ops
    assert "MemcpyToHost" not in all_ops
    assert "MemcpyFromHost" not in all_ops


def _assert_single_reshape_not_fused(events):
    ops_by_provider = _ops_by_provider(events)
    musa_ops = ops_by_provider.get("MUSAExecutionProvider", set())
    all_ops = set().union(*ops_by_provider.values())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert not fused_ops
    assert "Reshape" in musa_ops
    assert "MemcpyToHost" not in all_ops
    assert "MemcpyFromHost" not in all_ops


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
        helper.make_node("Reshape", ["Data2", "target_i64"], ["Z"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_metadata_reshape",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M", "K"]),
            helper.make_tensor_value_info("Data", TensorProto.FLOAT, ["N", "MK"]),
            helper.make_tensor_value_info("Data2", TensorProto.FLOAT, ["N", "MK"]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("Z", TensorProto.FLOAT, None),
        ],
        initializer=[gather_index, suffix],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_shape_pre_gather_metadata_reshape_model() -> bytes:
    pre_gather_index = numpy_helper.from_array(
        np.array([1, 0, 2], dtype=np.int64), name="pre_gather_index"
    )
    gather_index = numpy_helper.from_array(
        np.array([0, 1], dtype=np.int64), name="gather_index"
    )
    suffix = numpy_helper.from_array(np.array([4], dtype=np.int32), name="suffix")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node(
            "Gather", ["shape_i64", "pre_gather_index"], ["shape_perm"], axis=0
        ),
        helper.make_node("Cast", ["shape_perm"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Gather", ["shape_i32", "gather_index"], ["prefix"], axis=0),
        helper.make_node("Concat", ["prefix", "suffix"], ["target_i32"], axis=0),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["Data", "target_i64"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_pre_gather_metadata_reshape",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M", "K"]),
            helper.make_tensor_value_info("Data", TensorProto.FLOAT, ["M", "NK"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[pre_gather_index, gather_index, suffix],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_shape_reshape_with_zero_and_infer_dim_model() -> bytes:
    gather_index = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="gather_index"
    )
    suffix = numpy_helper.from_array(np.array([0, -1], dtype=np.int32), name="suffix")
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
        "shape_reshape_zero_and_infer_dim",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M", "K"]),
            helper.make_tensor_value_info("Data", TensorProto.FLOAT, ["N", "M", "K"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[gather_index, suffix],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_shape_reshape_fusion_multiple_reshape_consumers(tmp_path):
    model = _build_shape_metadata_reshape_model()
    x = np.zeros((2, 3, 4), dtype=np.float32)
    data = np.arange(24, dtype=np.float32).reshape(2, 12)
    data2 = data + 100.0
    feeds = {"X": x, "Data": data, "Data2": data2}

    expected = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    actual, events = _profile_musa_session(model, feeds, tmp_path, "shape_reshape")

    for actual_output, expected_output in zip(actual, expected):
        np.testing.assert_array_equal(actual_output, expected_output)
    _assert_shape_reshape_fused(events)


def test_single_reshape_after_pre_gather_is_not_fused(tmp_path):
    model = _build_shape_pre_gather_metadata_reshape_model()
    feeds = {
        "X": np.zeros((2, 3, 4), dtype=np.float32),
        "Data": np.arange(24, dtype=np.float32).reshape(3, 8),
    }

    expected = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    actual, events = _profile_musa_session(
        model, feeds, tmp_path, "shape_reshape_pre_gather"
    )

    for actual_output, expected_output in zip(actual, expected):
        np.testing.assert_array_equal(actual_output, expected_output)
    _assert_single_reshape_not_fused(events)


def test_single_reshape_with_zero_and_inferred_dim_is_not_fused(tmp_path):
    model = _build_shape_reshape_with_zero_and_infer_dim_model()
    feeds = {
        "X": np.zeros((2, 3, 4), dtype=np.float32),
        "Data": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }

    expected = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    actual, events = _profile_musa_session(
        model, feeds, tmp_path, "shape_reshape_zero_infer"
    )

    for actual_output, expected_output in zip(actual, expected):
        np.testing.assert_array_equal(actual_output, expected_output)
    _assert_single_reshape_not_fused(events)
