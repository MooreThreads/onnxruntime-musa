# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the MatMul operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_matmul_2d():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_batched():
    a = np.random.default_rng(2).standard_normal((16, 16, 32)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((16, 32, 64)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_batched_broadcast_left_batch():
    a = np.random.default_rng(4).standard_normal((1, 2, 3)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((4, 3, 5)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_rank4_batched():
    a = np.random.default_rng(6).standard_normal((2, 1, 3, 4)).astype(np.float32)
    b = np.random.default_rng(7).standard_normal((1, 5, 4, 6)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)], rtol=2e-3, atol=2e-4)
