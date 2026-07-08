# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Relu operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_relu_float():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Relu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_relu_float_5d():
    x = np.random.default_rng(1).standard_normal((2, 3, 1, 4, 5)).astype(np.float32)
    run_and_compare("Relu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_relu_float_matrix_with_negatives():
    x = np.array([[-3.5, -0.0, 2.0], [0.0, -7.25, 9.5]], dtype=np.float32)
    run_and_compare("Relu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
