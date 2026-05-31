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
