# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the GreaterOrEqual operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_greater_or_equal_float():
    a = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32)
    run_and_compare(
        "GreaterOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
    )


def test_greater_or_equal_broadcast():
    a = np.random.default_rng(2).standard_normal((8, 1)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 16)).astype(np.float32)
    run_and_compare(
        "GreaterOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
    )


def test_greater_or_equal_int64():
    a = np.arange(32, dtype=np.int64).reshape(4, 8)
    b = np.arange(31, -1, -1, dtype=np.int64).reshape(4, 8)
    run_and_compare(
        "GreaterOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
    )
