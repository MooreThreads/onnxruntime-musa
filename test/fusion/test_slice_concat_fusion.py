# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the Plugin EP Slice -> Concat fusion pattern."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, run_model_and_compare


def _int64_initializer(name, values):
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


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
