# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Reshape operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reshape_float():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    shape = np.array([32, 16], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_reshape_with_minus_one():
    x = np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32)
    shape = np.array([16, -1], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_reshape_int64():
    x = np.arange(4096, dtype=np.int64).reshape(16, 16, 16)
    shape = np.array([64, 64], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.INT64)],
    )
