# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Concat operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_concat_axis0():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_concat_axis1():
    a = np.random.default_rng(2).standard_normal((32, 16)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_int64_negative_axis():
    a = np.arange(512, dtype=np.int64).reshape(16, 32)
    b = np.arange(512, 1024, dtype=np.int64).reshape(16, 32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": -1},
    )
