# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceSumSquare operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reduce_sum_square_axis1_keepdims():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "ReduceSumSquare",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_sum_square_axis0_no_keepdims():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "ReduceSumSquare",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0], "keepdims": 0},
    )


def test_reduce_sum_square_int32_negative_axis_keepdims():
    x = np.arange(-12, 12, dtype=np.int32).reshape(2, 3, 4)
    run_and_compare(
        "ReduceSumSquare",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axes": [-1], "keepdims": 1},
    )


def test_reduce_sum_square_int64_axis0_no_keepdims():
    x = np.arange(-12, 12, dtype=np.int64).reshape(2, 3, 4)
    run_and_compare(
        "ReduceSumSquare",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axes": [0], "keepdims": 0},
    )


def test_reduce_sum_square_multi_axis_float_no_keepdims():
    x = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceSumSquare",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0, 2], "keepdims": 0},
    )
