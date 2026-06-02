# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Where operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_where_float():
    condition = np.array([True, False, True, False, True], dtype=bool)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    run_and_compare(
        "Where",
        inputs={"condition": condition, "X": x, "Y": y},
        outputs=[("Z", TensorProto.FLOAT)],
    )


def test_where_broadcast():
    condition = np.array([[True, False], [False, True]], dtype=bool)
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    run_and_compare(
        "Where",
        inputs={"condition": condition, "X": x, "Y": y},
        outputs=[("Z", TensorProto.FLOAT)],
    )


def test_where_int64():
    rng = np.random.default_rng(0)
    condition = rng.integers(0, 2, size=(4, 8)).astype(bool)
    x = rng.integers(0, 100, size=(4, 8)).astype(np.int64)
    y = rng.integers(-100, 0, size=(4, 8)).astype(np.int64)
    run_and_compare(
        "Where",
        inputs={"condition": condition, "X": x, "Y": y},
        outputs=[("Z", TensorProto.INT64)],
    )
