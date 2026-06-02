# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the And operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_and_basic():
    a = np.array([True, False, True, False], dtype=bool)
    b = np.array([True, True, False, False], dtype=bool)
    run_and_compare("And", inputs={"A": a, "B": b}, outputs=[("C", TensorProto.BOOL)])


def test_and_broadcast():
    a = np.array([[True], [False]], dtype=bool)
    b = np.array([True, False, True], dtype=bool)
    run_and_compare("And", inputs={"A": a, "B": b}, outputs=[("C", TensorProto.BOOL)])


def test_and_2d():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=(4, 8)).astype(bool)
    b = rng.integers(0, 2, size=(4, 8)).astype(bool)
    run_and_compare("And", inputs={"A": a, "B": b}, outputs=[("C", TensorProto.BOOL)])
