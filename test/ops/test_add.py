# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Add operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_add_float():
    a = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_add_int64():
    a = np.arange(-6, 6, dtype=np.int64).reshape(3, 4)
    b = np.arange(1, 13, dtype=np.int64).reshape(3, 4)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_add_broadcast():
    a = np.random.default_rng(2).standard_normal((2, 1, 4)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 3, 4)).astype(np.float32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
