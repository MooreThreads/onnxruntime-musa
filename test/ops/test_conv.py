# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Conv operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_conv_opset11_float_nchw():
    x = np.arange(1 * 1 * 5 * 4, dtype=np.float32).reshape(1, 1, 5, 4) / 10.0
    w = np.array([[[[1.0], [0.5], [-1.0]]]], dtype=np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"pads": [1, 0, 1, 0], "strides": [1, 1]},
        opset=11,
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv_lowered_conv1d_h1_k1():
    x = np.random.default_rng(0).standard_normal((2, 64, 1, 8)).astype(np.float32)
    w = np.random.default_rng(1).standard_normal((16, 64, 1, 1)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 1],
            "strides": [1, 1],
        },
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv_lowered_conv1d_h1_k1_with_bias():
    x = np.random.default_rng(2).standard_normal((2, 8, 1, 5)).astype(np.float32)
    w = np.random.default_rng(3).standard_normal((4, 8, 1, 1)).astype(np.float32)
    b = np.random.default_rng(4).standard_normal((4,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 1],
            "strides": [1, 1],
        },
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv_heightwise_width1_kernel_k5_pad2():
    x = np.random.default_rng(5).standard_normal((2, 1, 17, 8)).astype(np.float32)
    w = np.random.default_rng(6).standard_normal((1, 1, 5, 1)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [5, 1],
            "pads": [2, 0, 2, 0],
            "strides": [1, 1],
        },
        opset=13,
        rtol=1e-5,
        atol=1e-5,
    )
