# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Expand operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_expand_float():
    x = np.random.default_rng(0).standard_normal((1, 3, 1)).astype(np.float32)
    shape = np.array([2, 3, 4], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.FLOAT)])


def test_expand_bool():
    x = np.array([[True], [False]], dtype=np.bool_)
    shape = np.array([2, 3], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.BOOL)])
