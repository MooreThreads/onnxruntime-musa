# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the Plugin EP Concat -> MatMul fusion pattern."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run_model_and_compare


def test_concat_matmul_fusion_concat_on_lhs():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    x1 = rng.standard_normal((2, 3, 4, 7)).astype(np.float32)
    b = rng.standard_normal((2, 3, 12, 6)).astype(np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=-1),
        helper.make_node("MatMul", ["C", "B"], ["Y"]),
    ]
    feeds = {"X0": x0, "X1": x1, "B": b}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="concat_matmul_lhs_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_concat_matmul_fusion_concat_on_rhs():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 3, 4, 6)).astype(np.float32)
    x0 = rng.standard_normal((2, 3, 6, 5)).astype(np.float32)
    x1 = rng.standard_normal((2, 3, 6, 7)).astype(np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=-1),
        helper.make_node("MatMul", ["A", "C"], ["Y"]),
    ]
    feeds = {"A": a, "X0": x0, "X1": x1}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="concat_matmul_rhs_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_dynamic_concat_matmul_rank_broadcast_not_fused():
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((5, 3, 4)).astype(np.float32)
    x1 = rng.standard_normal((5, 3, 5)).astype(np.float32)
    b = rng.standard_normal((9, 6)).astype(np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=-1),
        helper.make_node("MatMul", ["C", "B"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dynamic_concat_matmul_rank_broadcast_graph",
        [
            helper.make_tensor_value_info(
                "X0", TensorProto.FLOAT, ["batch", 3, 4]
            ),
            helper.make_tensor_value_info(
                "X1", TensorProto.FLOAT, ["batch", 3, 5]
            ),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [9, 6]),
        ],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, ["batch", 3, 6]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X0": x0, "X1": x1, "B": b},
        rtol=1e-3,
        atol=1e-3,
    )
