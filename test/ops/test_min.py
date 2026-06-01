# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Min operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_min_float_binary_broadcast():
    a = np.random.default_rng(0).standard_normal((16, 1)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((1, 32)).astype(np.float32)
    run_and_compare("Min", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_min_int32_variadic():
    a = np.arange(32, dtype=np.int32).reshape(8, 4)
    b = np.full((8, 4), 7, dtype=np.int32)
    c = np.arange(31, -1, -1, dtype=np.int32).reshape(8, 4)
    run_and_compare("Min", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.INT32)])
