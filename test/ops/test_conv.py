# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Conv operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_conv_2d_basic():
    X = np.random.default_rng(0).standard_normal((1, 1, 5, 5)).astype(np.float32)
    W = np.random.default_rng(1).standard_normal((1, 1, 3, 3)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": X, "W": W},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"kernel_shape": [3, 3]},
        rtol=1e-4,
        atol=1e-5,
    )


def test_conv_2d_with_bias():
    X = np.random.default_rng(2).standard_normal((2, 3, 6, 6)).astype(np.float32)
    W = np.random.default_rng(3).standard_normal((4, 3, 3, 3)).astype(np.float32)
    B = np.random.default_rng(4).standard_normal((4,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": X, "W": W, "B": B},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"kernel_shape": [3, 3]},
        rtol=1e-4,
        atol=1e-5,
    )


def test_conv_2d_stride2():
    X = np.random.default_rng(5).standard_normal((1, 1, 8, 8)).astype(np.float32)
    W = np.random.default_rng(6).standard_normal((2, 1, 3, 3)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": X, "W": W},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"kernel_shape": [3, 3], "strides": [2, 2]},
        rtol=1e-4,
        atol=1e-5,
    )


def test_conv_2d_padded():
    X = np.random.default_rng(7).standard_normal((1, 1, 4, 4)).astype(np.float32)
    W = np.random.default_rng(8).standard_normal((1, 1, 3, 3)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": X, "W": W},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        rtol=1e-4,
        atol=1e-5,
    )
