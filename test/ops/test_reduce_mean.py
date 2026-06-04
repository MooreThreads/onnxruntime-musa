# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceMean operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_reduce_mean_axis1_keepdims():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_mean_axis0_no_keepdims():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0], "keepdims": 0},
    )


def test_reduce_mean_negative_axis_keepdims():
    x = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [-1], "keepdims": 1},
    )


def test_reduce_mean_axis2_no_keepdims_3d():
    x = np.random.default_rng(3).standard_normal((2, 3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [2], "keepdims": 0},
    )


def test_reduce_mean_multi_axis_keepdims():
    x = np.random.default_rng(4).standard_normal((2, 3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0, 2], "keepdims": 1},
    )


def test_reduce_mean_multi_axis_no_keepdims():
    x = np.random.default_rng(5).standard_normal((2, 3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0, 2], "keepdims": 0},
    )


def test_reduce_mean_float16_axis1_no_keepdims():
    x = np.random.default_rng(6).standard_normal((4, 8)).astype(np.float16)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axes": [1], "keepdims": 0},
        rtol=2e-2,
        atol=2e-2,
    )


def test_reduce_mean_double_axis0_keepdims():
    x = np.random.default_rng(7).standard_normal((4, 8)).astype(np.float64)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        attrs={"axes": [0], "keepdims": 1},
        rtol=1e-6,
        atol=1e-7,
    )


def test_reduce_mean_bfloat16_axis1_no_keepdims():
    x_f32 = np.random.default_rng(8).standard_normal((4, 8)).astype(np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    x_bf16_f32 = bfloat16_bits_to_float32(x)
    expected = np.mean(x_bf16_f32, axis=1)
    model = build_model_with_input_types(
        "ReduceMean",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"axes": [1], "keepdims": 0},
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_reduce_mean_int32_axis1_no_keepdims():
    x = np.array([[1, 2, 4], [5, 7, 9]], dtype=np.int32)
    run_and_compare(
        "ReduceMean",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axes": [1], "keepdims": 0},
    )
