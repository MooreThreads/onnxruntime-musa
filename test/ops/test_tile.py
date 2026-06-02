# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Tile operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_tile_1d():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    repeats = np.array([4], dtype=np.int64)
    run_and_compare(
        "Tile",
        inputs={"input": x, "repeats": repeats},
        outputs=[("output", TensorProto.FLOAT)],
    )


def test_tile_2d():
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    repeats = np.array([2, 3], dtype=np.int64)
    run_and_compare(
        "Tile",
        inputs={"input": x, "repeats": repeats},
        outputs=[("output", TensorProto.FLOAT)],
    )


def test_tile_int64():
    x = np.arange(6, dtype=np.int64).reshape(2, 3)
    repeats = np.array([3, 2], dtype=np.int64)
    run_and_compare(
        "Tile",
        inputs={"input": x, "repeats": repeats},
        outputs=[("output", TensorProto.INT64)],
    )
