# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the BatchNormalization operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    run_and_compare,
    run_with_iobinding,
)


def test_batch_normalization_float_2d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 4)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, (4,)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, (4,)).astype(np.float32)
    mean = rng.standard_normal((4,)).astype(np.float32)
    var = rng.uniform(0.5, 2.0, (4,)).astype(np.float32)
    run_and_compare(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"epsilon": 1e-5},
        opset=15,
    )


def test_batch_normalization_float_4d():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, (3,)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, (3,)).astype(np.float32)
    mean = rng.standard_normal((3,)).astype(np.float32)
    var = rng.uniform(0.5, 2.0, (3,)).astype(np.float32)
    run_and_compare(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"epsilon": 1e-5},
        opset=15,
    )


def test_batch_normalization_opset13_float_4d():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, (3,)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, (3,)).astype(np.float32)
    mean = rng.standard_normal((3,)).astype(np.float32)
    var = rng.uniform(0.5, 2.0, (3,)).astype(np.float32)
    run_and_compare(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"epsilon": 1e-3},
        opset=13,
    )


def test_batch_normalization_float_3d_custom_epsilon():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 4, 6)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, (4,)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, (4,)).astype(np.float32)
    mean = rng.standard_normal((4,)).astype(np.float32)
    var = rng.uniform(0.5, 2.0, (4,)).astype(np.float32)
    run_and_compare(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"epsilon": 1e-3},
        opset=15,
    )


def test_batch_normalization_double_2d():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 3)).astype(np.float64)
    scale = rng.uniform(0.5, 1.5, (3,)).astype(np.float64)
    bias = rng.uniform(-0.2, 0.2, (3,)).astype(np.float64)
    mean = rng.standard_normal((3,)).astype(np.float64)
    var = rng.uniform(0.5, 2.0, (3,)).astype(np.float64)
    run_and_compare(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        outputs=[("Y", TensorProto.DOUBLE)],
        attrs={"epsilon": 1e-5},
        opset=15,
        rtol=1e-9,
        atol=1e-10,
    )


def test_batch_normalization_float16_4d():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 2, 2)).astype(np.float16)
    scale = rng.uniform(0.5, 1.5, (3,)).astype(np.float16)
    bias = rng.uniform(-0.2, 0.2, (3,)).astype(np.float16)
    mean = rng.standard_normal((3,)).astype(np.float16)
    var = rng.uniform(0.5, 2.0, (3,)).astype(np.float16)
    epsilon = 1e-3
    expected = (
        (x.astype(np.float32) - mean.astype(np.float32).reshape(1, 3, 1, 1))
        / np.sqrt(var.astype(np.float32).reshape(1, 3, 1, 1) + epsilon)
        * scale.astype(np.float32).reshape(1, 3, 1, 1)
        + bias.astype(np.float32).reshape(1, 3, 1, 1)
    ).astype(np.float16)
    model = build_model_with_input_types(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        input_types={
            "X": TensorProto.FLOAT16,
            "scale": TensorProto.FLOAT16,
            "B": TensorProto.FLOAT16,
            "input_mean": TensorProto.FLOAT16,
            "input_var": TensorProto.FLOAT16,
        },
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"epsilon": epsilon},
        opset=15,
    )
    (actual,) = run_with_iobinding(
        model,
        {
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        {
            "X": TensorProto.FLOAT16,
            "scale": TensorProto.FLOAT16,
            "B": TensorProto.FLOAT16,
            "input_mean": TensorProto.FLOAT16,
            "input_var": TensorProto.FLOAT16,
        },
        [("Y", TensorProto.FLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
