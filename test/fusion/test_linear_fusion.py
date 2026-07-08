# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for linear Plugin EP fusion patterns."""

import numpy as np
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, build_graph_model, run_model_and_compare


def test_gemm_relu_fusion():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    c = rng.standard_normal((12,)).astype(np.float32)

    nodes = [
        helper.make_node("Gemm", ["A", "B", "C"], ["G"], alpha=1.0, beta=1.0),
        helper.make_node("Relu", ["G"], ["Y"]),
    ]
    feeds = {"A": a, "B": b, "C": c}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="gemm_relu_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_matmul_add_tanh_fusion():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 4, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    bias = rng.standard_normal((12,)).astype(np.float32)

    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["M"]),
        helper.make_node("Add", ["M", "Bias"], ["MB"]),
        helper.make_node("Tanh", ["MB"], ["Y"]),
    ]
    feeds = {"A": a, "B": b, "Bias": bias}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="matmul_add_tanh_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_matmul_add_tanh_fusion_with_initializer_inputs():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((2, 4, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    bias = rng.standard_normal((12,)).astype(np.float32)

    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["M"]),
        helper.make_node("Add", ["M", "Bias"], ["MB"]),
        helper.make_node("Tanh", ["MB"], ["Y"]),
    ]
    feeds = {"A": a}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(bias, "Bias"),
        ],
        name="matmul_add_tanh_initializer_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
