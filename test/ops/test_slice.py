# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Slice operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_slice_2d():
    data = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)
    starts = np.array([0, 1], dtype=np.int64)
    ends = np.array([3, 4], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes, "steps": steps},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_slice_with_step():
    data = np.random.default_rng(1).standard_normal((6,)).astype(np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([6], dtype=np.int64)
    axes = np.array([0], dtype=np.int64)
    steps = np.array([2], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes, "steps": steps},
        outputs=[("Y", TensorProto.FLOAT)],
    )
