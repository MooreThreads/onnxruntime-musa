# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Sqrt operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_sqrt_float():
    x = np.random.default_rng(0).uniform(0.0, 10.0, (16, 32)).astype(np.float32)
    run_and_compare("Sqrt", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_sqrt_float_matrix_with_zero():
    x = np.array([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=np.float32)
    run_and_compare("Sqrt", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
