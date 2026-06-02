# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the IsNaN operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_is_nan_no_nans():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    run_and_compare("IsNaN", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])


def test_is_nan_with_nans():
    x = np.array([1.0, float("nan"), 0.0, float("nan"), -1.0], dtype=np.float32)
    run_and_compare("IsNaN", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])


def test_is_nan_2d():
    x = np.array([[1.0, float("nan")], [float("nan"), 2.0]], dtype=np.float32)
    run_and_compare("IsNaN", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])
