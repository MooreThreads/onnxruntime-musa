# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Exp operator."""

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


def test_exp_float():
    x = np.array([[-1.0, 0.0, 1.0, 3.0]], dtype=np.float32)
    run_and_compare("Exp", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
        (np.float64, TensorProto.DOUBLE, 1e-6, 1e-7),
    ],
)
def test_exp_float_like_dtypes(np_dtype, tensor_type, rtol, atol):
    x = np.array([[-1.0, 0.0, 1.0, 3.0]], dtype=np_dtype)
    run_and_compare(
        "Exp",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        rtol=rtol,
        atol=atol,
    )


def test_exp_bfloat16():
    x_f32 = np.array([[-1.0, 0.0, 1.0, 3.0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    expected = np.exp(bfloat16_bits_to_float32(x))
    model = build_model_with_input_types(
        "Exp",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )
