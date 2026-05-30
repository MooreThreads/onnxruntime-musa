# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Sub operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_sub_float():
    a = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    run_and_compare("Sub", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_sub_int64():
    a = np.arange(10, 22, dtype=np.int64).reshape(3, 4)
    b = np.arange(1, 13, dtype=np.int64).reshape(3, 4)
    run_and_compare("Sub", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_sub_broadcast():
    a = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((4,)).astype(np.float32)
    run_and_compare("Sub", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
