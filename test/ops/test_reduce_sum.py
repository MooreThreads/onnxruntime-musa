# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceSum operator.

Since opset 13, ReduceSum takes `axes` as a (second) input rather than an
attribute.
"""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


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


def test_reduce_sum_int32_negative_axis_no_keepdims():
    x = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
    axes = np.array([-1], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"keepdims": 0},
    )


def test_reduce_sum_float_axis0_keepdims_3d():
    x = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32)
    axes = np.array([0], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"keepdims": 1},
    )


def test_reduce_sum_multi_axis_int32_no_keepdims():
    x = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
    axes = np.array([0, 2], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"keepdims": 0},
    )


def test_reduce_sum_float16_axis1_no_keepdims():
    x = np.random.default_rng(3).standard_normal((4, 8)).astype(np.float16)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"keepdims": 0},
        rtol=2e-2,
        atol=2e-2,
    )


def test_reduce_sum_double_axis0_keepdims():
    x = np.random.default_rng(4).standard_normal((4, 8)).astype(np.float64)
    axes = np.array([0], dtype=np.int64)
    run_and_compare(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.DOUBLE)],
        attrs={"keepdims": 1},
        rtol=1e-6,
        atol=1e-7,
    )


def test_reduce_sum_bfloat16_axis1_no_keepdims():
    x_f32 = np.random.default_rng(5).standard_normal((4, 8)).astype(np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    x_bf16_f32 = bfloat16_bits_to_float32(x)
    axes = np.array([1], dtype=np.int64)
    expected = np.sum(x_bf16_f32, axis=1)
    model = build_model_with_input_types(
        "ReduceSum",
        inputs={"X": x, "axes": axes},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"keepdims": 0},
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x, "axes": axes},
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
