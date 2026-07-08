# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Sigmoid operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_sigmoid_float():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Sigmoid", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_sigmoid_float_matrix_extremes():
    x = np.array([[-8.0, -1.0, 0.0], [1.0, 8.0, 16.0]], dtype=np.float32)
    run_and_compare("Sigmoid", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
