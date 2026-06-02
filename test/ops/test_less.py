# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Less operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_less_float():
    a = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32)
    run_and_compare("Less", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_less_broadcast():
    a = np.random.default_rng(2).standard_normal((8, 1)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 16)).astype(np.float32)
    run_and_compare("Less", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_less_int32():
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    b = np.arange(15, -1, -1, dtype=np.int32).reshape(4, 4)
    run_and_compare("Less", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])
