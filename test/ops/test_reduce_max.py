# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceMax operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reduce_max_axis1_keepdims():
    x = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
    run_and_compare(
        "ReduceMax",
        inputs={"data": x},
        outputs=[("reduced", TensorProto.FLOAT)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_max_axis0_no_keepdims():
    x = np.random.default_rng(1).standard_normal((8, 16)).astype(np.float32)
    run_and_compare(
        "ReduceMax",
        inputs={"data": x},
        outputs=[("reduced", TensorProto.FLOAT)],
        attrs={"axes": [0], "keepdims": 0},
    )


def test_reduce_max_int64():
    x = np.arange(24, dtype=np.int64).reshape(4, 6)
    run_and_compare(
        "ReduceMax",
        inputs={"data": x},
        outputs=[("reduced", TensorProto.INT64)],
        attrs={"axes": [1], "keepdims": 0},
    )
