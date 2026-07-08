# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for PRelu."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_prelu_float_channel_slope():
    x = np.random.default_rng(0).standard_normal((5, 64)).astype(np.float32)
    slope = np.linspace(0.01, 0.25, num=64, dtype=np.float32)
    run_and_compare(
        "PRelu",
        inputs={"X": x, "slope": slope},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=17,
        rtol=1e-5,
        atol=1e-6,
    )


def test_prelu_float_scalar_slope():
    x = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    slope = np.array(0.2, dtype=np.float32)
    run_and_compare(
        "PRelu",
        inputs={"X": x, "slope": slope},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=17,
        rtol=1e-5,
        atol=1e-6,
    )
