# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for LessOrEqual."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_less_or_equal_int64_broadcast():
    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    b = np.array([0, 3, 4], dtype=np.int64)
    run_and_compare(
        "LessOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=17,
        rtol=0,
        atol=0,
    )


def test_less_or_equal_float_scalar():
    a = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)
    b = np.array(0.0, dtype=np.float32)
    run_and_compare(
        "LessOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=17,
        rtol=0,
        atol=0,
    )
