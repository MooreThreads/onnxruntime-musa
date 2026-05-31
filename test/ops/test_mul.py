# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Mul operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_mul_float():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_mul_int64():
    a = np.arange(-256, 256, dtype=np.int64).reshape(16, 32)
    b = np.arange(1, 513, dtype=np.int64).reshape(16, 32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_mul_broadcast():
    a = np.random.default_rng(2).standard_normal((32, 16, 1)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 1, 32)).astype(np.float32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
