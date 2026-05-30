# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceMean operator.

At opset 13-17 ReduceMean takes `axes` as an attribute.
"""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reduce_mean_axis1_keepdims():
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_mean_axis0_no_keepdims():
    x = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0], "keepdims": 0},
    )
