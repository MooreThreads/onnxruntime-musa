# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceProd operator.

At opset 13-17 ReduceProd takes `axes` as an attribute.
"""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reduce_prod_axis1_keepdims():
    x = np.random.default_rng(0).uniform(0.5, 1.5, (3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_prod_int64():
    x = np.arange(1, 13, dtype=np.int64).reshape(3, 4)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axes": [1], "keepdims": 0},
    )
