# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceSum operator.

Since opset 13, ReduceSum takes `axes` as a (second) input rather than an
attribute.
"""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reduce_sum_axis1_keepdims():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"keepdims": 1},
    )


def test_reduce_sum_axis0_no_keepdims():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    axes = np.array([0], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"keepdims": 0},
    )


def test_reduce_sum_int64():
    x = np.arange(512, dtype=np.int64).reshape(16, 32)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"keepdims": 1},
    )
