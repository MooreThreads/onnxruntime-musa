# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Add operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_add_float():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_add_int64():
    a = np.arange(-256, 256, dtype=np.int64).reshape(16, 32)
    b = np.arange(1, 513, dtype=np.int64).reshape(16, 32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_add_broadcast():
    a = np.random.default_rng(2).standard_normal((32, 1, 16)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 32, 16)).astype(np.float32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_add_float_5d_broadcast():
    a = np.random.default_rng(4).standard_normal((2, 1, 3, 1, 4)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((1, 5, 1, 6, 4)).astype(np.float32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_add_int32_scalar_broadcast():
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    b = np.array(5, dtype=np.int32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT32)])


def test_add_int64_multidirectional_broadcast():
    a = np.arange(2 * 3 * 1, dtype=np.int64).reshape(2, 3, 1)
    b = np.array([10, 20, 30, 40], dtype=np.int64).reshape(1, 1, 4)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])
