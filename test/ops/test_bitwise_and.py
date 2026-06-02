# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the BitwiseAnd operator (opset 18)."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_bitwise_and_int32():
    a = np.array([0b1100, 0b1010, 0b1111], dtype=np.int32)
    b = np.array([0b1010, 0b1010, 0b0101], dtype=np.int32)
    run_and_compare(
        "BitwiseAnd",
        inputs={"A": a, "B": b},
        outputs=[("C", TensorProto.INT32)],
        opset=18,
    )


def test_bitwise_and_int64():
    a = np.arange(16, dtype=np.int64).reshape(4, 4)
    b = np.array([0xFF, 0x0F, 0xF0, 0x00], dtype=np.int64)
    run_and_compare(
        "BitwiseAnd",
        inputs={"A": a, "B": b},
        outputs=[("C", TensorProto.INT64)],
        opset=18,
    )
