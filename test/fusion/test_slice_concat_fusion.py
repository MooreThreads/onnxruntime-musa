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
"""End-to-end tests for the Plugin EP Slice -> Concat fusion pattern."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    musa_devices,
    run_model_and_compare,
)


def _int64_initializer(name, values):
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def _profile_musa_ops(model, feeds, tmp_path, prefix):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    session.run(None, dict(feeds))
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)

    ops = set()
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        args = event.get("args", {})
        if args.get("provider") == "MUSAExecutionProvider":
            ops.add(args.get("op_name"))
    return ops


def test_slice_concat_fusion_rank2_column_slices_dynamic_batch():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 20)).astype(np.float32)

    nodes = []
    initializers = []
    bounds = [(0, 2), (2, 5), (5, 6), (6, 9),
              (9, 11), (11, 14), (14, 18), (18, 20)]
    concat_inputs = []
    for i, (start, end) in enumerate(bounds):
        starts = f"starts_{i}"
        ends = f"ends_{i}"
        axes = f"axes_{i}"
        sliced = f"S{i}"
        initializers.extend([
            _int64_initializer(starts, [0, start]),
            _int64_initializer(ends, [9223372036854775807, end]),
            _int64_initializer(axes, [0, 1]),
        ])
        nodes.append(helper.make_node("Slice", ["X", starts, ends, axes], [sliced]))
        concat_inputs.append(sliced)

    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=1))
    graph = helper.make_graph(
        nodes,
        "slice_concat_fusion_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 20])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 20])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X": x},
        rtol=1e-5,
        atol=1e-5,
    )


def test_slice_concat_axis0_is_not_misclassified():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 3)).astype(np.float32)

    initializers = [
        _int64_initializer("starts", [0, 0]),
        _int64_initializer("ends", [2, 3]),
        _int64_initializer("axes", [0, 1]),
    ]
    nodes = [
        helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["S"]),
        helper.make_node("Concat", ["S", "S"], ["Y"], axis=0),
    ]
    model = build_graph_model(
        nodes,
        {"X": x},
        [("Y", TensorProto.FLOAT)],
        initializers=initializers,
        name="slice_concat_axis0_not_fusion_graph",
    )

    run_model_and_compare(
        model,
        {"X": x},
        rtol=1e-5,
        atol=1e-5,
    )


def test_slice_concat_fusion_accepts_negative_column_axis():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((3, 10)).astype(np.float32)

    nodes = []
    initializers = []
    concat_inputs = []
    for i, (start, end) in enumerate([(0, 3), (3, 7), (7, 10)]):
        starts = f"starts_{i}"
        ends = f"ends_{i}"
        axes = f"axes_{i}"
        sliced = f"S{i}"
        initializers.extend([
            _int64_initializer(starts, [0, start]),
            _int64_initializer(ends, [9223372036854775807, end]),
            _int64_initializer(axes, [0, 1]),
        ])
        nodes.append(helper.make_node("Slice", ["X", starts, ends, axes], [sliced]))
        concat_inputs.append(sliced)

    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=-1))
    model = build_graph_model(
        nodes,
        {"X": x},
        [("Y", TensorProto.FLOAT)],
        initializers=initializers,
        name="slice_concat_negative_axis_fusion_graph",
    )

    run_model_and_compare(
        model,
        {"X": x},
        rtol=1e-5,
        atol=1e-5,
    )


def test_slice_concat_fusion_many_single_column_slices():
    rng = np.random.default_rng(7)
    cols = 64
    x = rng.standard_normal((4, cols)).astype(np.float32)

    nodes = []
    initializers = []
    concat_inputs = []
    for i in range(cols):
        starts = f"starts_{i}"
        ends = f"ends_{i}"
        axes = f"axes_{i}"
        sliced = f"S{i}"
        initializers.extend([
            _int64_initializer(starts, [0, i]),
            _int64_initializer(ends, [9223372036854775807, i + 1]),
            _int64_initializer(axes, [0, 1]),
        ])
        nodes.append(helper.make_node("Slice", ["X", starts, ends, axes], [sliced]))
        concat_inputs.append(sliced)

    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=1))
    graph = helper.make_graph(
        nodes,
        "slice_concat_many_segments",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", cols])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", cols])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X": x},
        rtol=1e-5,
        atol=1e-5,
    )


def test_slice_concat_fusion_allows_zero_constant_of_shape_segment():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 20)).astype(np.float32)

    nodes = []
    initializers = [_int64_initializer("zero_shape", [3, 2])]
    value_info = [
        helper.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 2]),
    ]
    concat_inputs = []
    bounds = [(0, 8), (8, 16)]
    for i, (start, end) in enumerate(bounds):
        starts = f"starts_{i}"
        ends = f"ends_{i}"
        axes = f"axes_{i}"
        sliced = f"S{i}"
        initializers.extend([
            _int64_initializer(starts, [0, start]),
            _int64_initializer(ends, [9223372036854775807, end]),
            _int64_initializer(axes, [0, 1]),
        ])
        nodes.append(helper.make_node("Slice", ["X", starts, ends, axes], [sliced]))
        concat_inputs.append(sliced)
        if i == 0:
            nodes.append(
                helper.make_node(
                    "ConstantOfShape",
                    ["zero_shape"],
                    ["Z"],
                    value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
                )
            )
            concat_inputs.append("Z")

    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=1))
    graph = helper.make_graph(
        nodes,
        "slice_concat_zero_segment",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 20])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 18])],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X": x},
        rtol=1e-5,
        atol=1e-5,
    )


def test_slice_concat_fusion_allows_direct_rank2_inputs():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    d = rng.standard_normal((4, 4)).astype(np.float32)

    nodes = []
    initializers = []
    concat_inputs = ["D"]
    for i in range(8):
        starts = f"starts_{i}"
        ends = f"ends_{i}"
        axes = f"axes_{i}"
        sliced = f"S{i}"
        initializers.extend([
            _int64_initializer(starts, [0, i * 4]),
            _int64_initializer(ends, [9223372036854775807, (i + 1) * 4]),
            _int64_initializer(axes, [0, 1]),
        ])
        nodes.append(helper.make_node("Slice", ["X", starts, ends, axes], [sliced]))
        concat_inputs.append(sliced)

    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=1))
    graph = helper.make_graph(
        nodes,
        "slice_concat_direct_input",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 32]),
            helper.make_tensor_value_info("D", TensorProto.FLOAT, ["batch", 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 36])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X": x, "D": d},
        rtol=1e-5,
        atol=1e-5,
    )


def test_slice_concat_fusion_three_segments_fuses(tmp_path):
    rng = np.random.default_rng(19)
    x = rng.standard_normal((5, 6)).astype(np.float32)

    nodes = []
    initializers = []
    concat_inputs = []
    for i, (start, end) in enumerate([(0, 2), (2, 4), (4, 6)]):
        starts = f"starts_{i}"
        ends = f"ends_{i}"
        axes = f"axes_{i}"
        sliced = f"S{i}"
        initializers.extend([
            _int64_initializer(starts, [0, start]),
            _int64_initializer(ends, [9223372036854775807, end]),
            _int64_initializer(axes, [0, 1]),
        ])
        nodes.append(helper.make_node("Slice", ["X", starts, ends, axes], [sliced]))
        concat_inputs.append(sliced)

    nodes.append(helper.make_node("Concat", concat_inputs, ["C"], axis=1))
    nodes.append(helper.make_node("Relu", ["C"], ["Y"]))
    graph = helper.make_graph(
        nodes,
        "slice_concat_three_segments_fusion_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 6])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 6])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    model_bytes = model.SerializeToString()
    feeds = {"X": x}

    run_model_and_compare(model_bytes, feeds, rtol=1e-5, atol=1e-5)
    musa_ops = _profile_musa_ops(model_bytes, feeds, tmp_path, "slice_concat_three")
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "Slice" not in musa_ops
    assert "Concat" not in musa_ops
