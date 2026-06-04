# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Opset 19 smoke tests for reduction device kernels."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_opset19_reduce_mean_float16():
    x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float16)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"keepdims": 0},
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


def test_opset19_reduce_prod_int32():
    x = np.tile(np.array([1, 2, 3, 1], dtype=np.int32), (4, 1))
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"keepdims": 0},
        opset=19,
    )


def test_opset19_reduce_sum_double():
    x = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float64)
    axes = np.array([0], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.DOUBLE)],
        attrs={"keepdims": 1},
        opset=19,
        rtol=1e-6,
        atol=1e-7,
    )


def test_opset19_reduce_sum_square_float16():
    x = np.random.default_rng(2).standard_normal((4, 8)).astype(np.float16)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "ReduceSumSquare",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"keepdims": 0},
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )
