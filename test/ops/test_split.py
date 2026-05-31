# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Split operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_split_even_axis0():
    x = np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32)
    split = np.array([16, 16], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_split_uneven_axis1():
    x = np.random.default_rng(1).standard_normal((32, 48)).astype(np.float32)
    split = np.array([16, 32], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )
