# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Reshape operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reshape_float():
    x = np.random.default_rng(0).standard_normal((2, 6)).astype(np.float32)
    shape = np.array([3, 4], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_reshape_with_minus_one():
    x = np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)
    shape = np.array([2, -1], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_reshape_int64():
    x = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    shape = np.array([6, 4], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.INT64)],
    )
