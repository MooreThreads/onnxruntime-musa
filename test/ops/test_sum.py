# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Sum (variadic) operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_sum_two_inputs_same_shape():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_three_inputs():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)
    c = rng.standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_many_inputs_same_shape():
    rng = np.random.default_rng(3)
    inputs = {
        f"X{i}": rng.standard_normal((8, 16)).astype(np.float32)
        for i in range(10)
    }
    run_and_compare("Sum", inputs=inputs, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_broadcast():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((32,)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_three_inputs_broadcast():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((16, 1)).astype(np.float32)
    b = rng.standard_normal((1, 32)).astype(np.float32)
    c = rng.standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_three_inputs_broadcast_no_output_shape_input():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((16, 1, 1)).astype(np.float32)
    b = rng.standard_normal((1, 32, 1)).astype(np.float32)
    c = rng.standard_normal((1, 1, 8)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_float_three_inputs_multidirectional_broadcast():
    a = np.random.default_rng(2).standard_normal((2, 3, 1)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 1, 4)).astype(np.float32)
    c = np.random.default_rng(4).standard_normal((1, 3, 1)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_float_scalar_broadcast():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array(100.0, dtype=np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
