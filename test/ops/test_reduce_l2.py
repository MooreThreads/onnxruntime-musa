# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ReduceL2 operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
        (np.float32, TensorProto.FLOAT, 1e-5, 1e-6),
        (np.float64, TensorProto.DOUBLE, 1e-12, 1e-12),
        (np.int32, TensorProto.INT32, 0, 0),
    ],
)
def test_reduce_l2_opset13_registered_dtypes(np_dtype, tensor_type, rtol, atol):
    x = np.array([[1, 2, 2], [3, 4, 0]], dtype=np_dtype)
    run_and_compare(
        "ReduceL2",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        attrs={"axes": [1], "keepdims": 0},
        opset=13,
        rtol=rtol,
        atol=atol,
    )


def test_reduce_l2_float16_axis0_no_keepdims():
    x = np.random.default_rng(1).standard_normal((4, 5)).astype(np.float16)
    run_and_compare(
        "ReduceL2",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axes": [0], "keepdims": 0},
        rtol=2e-2,
        atol=2e-2,
    )


def test_reduce_l2_bfloat16_opset18():
    x_f32 = np.array([[1, 2, 2], [3, 4, 0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    axes = np.array([1], dtype=np.int64)
    expected = np.linalg.norm(bfloat16_bits_to_float32(x), axis=1)
    model = build_model_with_input_types(
        "ReduceL2",
        inputs={"X": x, "axes": axes},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"keepdims": 0},
        opset=18,
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
