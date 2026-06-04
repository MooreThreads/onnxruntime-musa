# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceProd operator.

At opset 13-17 ReduceProd takes `axes` as an attribute.
"""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reduce_prod_axis1_keepdims():
    x = np.random.default_rng(0).uniform(0.5, 1.5, (16, 32)).astype(np.float32)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_prod_int64():
    # Use tile of [1, 2] so each row product = 2**16 (no overflow)
    x = np.tile(np.array([1, 2], dtype=np.int64), (16, 16))
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axes": [1], "keepdims": 0},
    )


def test_reduce_prod_int32_negative_axis_keepdims():
    x = np.tile(np.array([1, 2, 3], dtype=np.int32), (2, 4, 1))
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axes": [-1], "keepdims": 1},
    )


def test_reduce_prod_float_axis0_no_keepdims_3d():
    x = np.random.default_rng(2).uniform(0.5, 1.25, (2, 3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0], "keepdims": 0},
    )


def test_reduce_prod_multi_axis_int32_no_keepdims():
    x = np.ones((2, 3, 4), dtype=np.int32)
    x[:, :, 0] = 2
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axes": [0, 2], "keepdims": 0},
    )
