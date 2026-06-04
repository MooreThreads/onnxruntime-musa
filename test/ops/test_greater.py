# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Greater operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_greater_float_broadcast():
    a = np.random.default_rng(0).standard_normal((16, 1)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((1, 32)).astype(np.float32)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_greater_int64():
    a = np.arange(32, dtype=np.int64).reshape(8, 4)
    b = np.arange(31, -1, -1, dtype=np.int64).reshape(8, 4)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_greater_int32_scalar_broadcast():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array(3, dtype=np.int32)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_greater_float_multidirectional_broadcast():
    a = np.array([0.0, 2.0, 4.0], dtype=np.float32).reshape(1, 3, 1)
    b = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32).reshape(1, 1, 4)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])
