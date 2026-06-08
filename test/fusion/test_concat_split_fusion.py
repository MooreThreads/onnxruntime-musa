# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the Plugin EP Concat -> Split fusion pattern."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, run_model_and_compare


def test_concat_split_fusion_rank2_axis1_internal_splits():
    rng = np.random.default_rng(6)
    x0 = rng.standard_normal((3, 5)).astype(np.float32)
    x1 = rng.standard_normal((3, 7)).astype(np.float32)
    x2 = rng.standard_normal((3, 4)).astype(np.float32)

    split_sizes = helper.make_tensor(
        "split_sizes", TensorProto.INT64, [7], [2, 3, 1, 6, 2, 1, 1]
    )
    nodes = [
        helper.make_node("Concat", ["X0", "X1", "X2"], ["Packed"], axis=1),
        helper.make_node(
            "Split",
            ["Packed", "split_sizes"],
            ["Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6"],
            axis=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "concat_split_fusion_graph",
        [
            helper.make_tensor_value_info("X0", TensorProto.FLOAT, ["batch", 5]),
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, ["batch", 7]),
            helper.make_tensor_value_info("X2", TensorProto.FLOAT, ["batch", 4]),
        ],
        [
            helper.make_tensor_value_info("Y0", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("Y1", TensorProto.FLOAT, ["batch", 3]),
            helper.make_tensor_value_info("Y2", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("Y3", TensorProto.FLOAT, ["batch", 6]),
            helper.make_tensor_value_info("Y4", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("Y5", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("Y6", TensorProto.FLOAT, ["batch", 1]),
        ],
        initializer=[split_sizes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X0": x0, "X1": x1, "X2": x2},
        rtol=1e-6,
        atol=1e-6,
    )


def test_concat_split_downstream_concat_fusion():
    rng = np.random.default_rng(8)
    x0 = rng.standard_normal((3, 4)).astype(np.float32)
    x1 = rng.standard_normal((3, 5)).astype(np.float32)
    x2 = rng.standard_normal((3, 3)).astype(np.float32)

    split_sizes = helper.make_tensor(
        "split_sizes", TensorProto.INT64, [4], [2, 2, 5, 3]
    )
    nodes = [
        helper.make_node("Concat", ["X0", "X1", "X2"], ["Packed"], axis=1),
        helper.make_node(
            "Split",
            ["Packed", "split_sizes"],
            ["S0", "S1", "S2", "S3"],
            axis=1,
        ),
        helper.make_node("Concat", ["S1", "S3", "S0"], ["Y"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "concat_split_downstream_concat_fusion_graph",
        [
            helper.make_tensor_value_info("X0", TensorProto.FLOAT, ["batch", 4]),
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, ["batch", 5]),
            helper.make_tensor_value_info("X2", TensorProto.FLOAT, ["batch", 3]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 7]),
            helper.make_tensor_value_info("S2", TensorProto.FLOAT, ["batch", 5]),
        ],
        initializer=[split_sizes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X0": x0, "X1": x1, "X2": x2},
        rtol=1e-6,
        atol=1e-6,
    )


def test_concat_split_downstream_sum_fusion():
    rng = np.random.default_rng(19)
    x0 = rng.standard_normal((4, 6)).astype(np.float32)
    x1 = rng.standard_normal((4, 4)).astype(np.float32)

    split_sizes = helper.make_tensor(
        "split_sizes", TensorProto.INT64, [5], [2, 2, 2, 2, 2]
    )
    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["Packed"], axis=1),
        helper.make_node(
            "Split",
            ["Packed", "split_sizes"],
            ["S0", "S1", "S2", "S3", "S4"],
            axis=1,
        ),
        helper.make_node("Sum", ["S0", "S1", "S2"], ["Ysum"]),
        helper.make_node("Concat", ["S3", "S4", "S0"], ["Ycat"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "concat_split_downstream_sum_fusion_graph",
        [
            helper.make_tensor_value_info("X0", TensorProto.FLOAT, ["batch", 6]),
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, ["batch", 4]),
        ],
        [
            helper.make_tensor_value_info("Ysum", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("Ycat", TensorProto.FLOAT, ["batch", 6]),
            helper.make_tensor_value_info("S2", TensorProto.FLOAT, ["batch", 2]),
        ],
        initializer=[split_sizes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X0": x0, "X1": x1},
        rtol=1e-6,
        atol=1e-6,
    )
