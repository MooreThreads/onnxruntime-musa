# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the NonZero operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_non_zero_1d():
    x = np.array([0, 1, 0, 3, -1], dtype=np.float32)
    run_and_compare("NonZero", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_non_zero_2d():
    x = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32)
    run_and_compare("NonZero", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_non_zero_bool():
    x = np.array([True, False, True, False, True], dtype=bool)
    run_and_compare("NonZero", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])
