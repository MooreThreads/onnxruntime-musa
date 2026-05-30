# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Div operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_div_float():
    a = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    b = np.random.default_rng(1).uniform(1.0, 2.0, (3, 4)).astype(np.float32)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_div_int64():
    # Exact divisors to avoid truncation-direction ambiguity between EPs.
    a = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64).reshape(2, 3)
    b = np.array([2, 4, 5, 8, 10, 12], dtype=np.int64).reshape(2, 3)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_div_broadcast():
    a = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32)
    b = np.random.default_rng(3).uniform(1.0, 2.0, (4,)).astype(np.float32)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
