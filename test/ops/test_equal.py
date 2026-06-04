# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Equal operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_equal_float_broadcast():
    a = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    b = np.array([[0.0, 2.0, 2.0]], dtype=np.float32)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_equal_bool():
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    b = np.array([[True, True], [False, False]], dtype=np.bool_)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_equal_int32_scalar_broadcast():
    a = np.array([[1, 2, 1], [3, 1, 4]], dtype=np.int32)
    b = np.array(1, dtype=np.int32)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_equal_int64_multidirectional_broadcast():
    a = np.array([1, 2, 3], dtype=np.int64).reshape(1, 3, 1)
    b = np.array([1, 0, 3, 4], dtype=np.int64).reshape(1, 1, 4)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])
