# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Neg operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_neg_float():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Neg", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_neg_float_matrix_with_zero():
    x = np.array([[-3.5, -0.0, 2.0], [0.0, -7.25, 9.5]], dtype=np.float32)
    run_and_compare("Neg", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_neg_int32():
    x = np.array([[-3, 0, 2], [4, -7, 9]], dtype=np.int32)
    run_and_compare("Neg", inputs={"X": x}, outputs=[("Y", TensorProto.INT32)])


def test_neg_float16():
    x = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float16)
    run_and_compare(
        "Neg",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        rtol=2e-2,
        atol=2e-2,
    )
