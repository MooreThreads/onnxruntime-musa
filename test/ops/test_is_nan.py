# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the IsNaN operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_is_nan_float():
    x = np.array([[-1.0, 0.0, np.nan, 3.0]], dtype=np.float32)
    run_and_compare("IsNaN", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])


@pytest.mark.parametrize("np_dtype", [np.float16, np.float64])
def test_is_nan_float_like_dtypes(np_dtype):
    x = np.array([[-1.0, 0.0, np.nan, 3.0]], dtype=np_dtype)
    run_and_compare("IsNaN", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])


def test_is_nan_bfloat16():
    x_f32 = np.array([[-1.0, 0.0, np.nan, 3.0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    model = build_model_with_input_types(
        "IsNaN",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BOOL)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BOOL, x.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, np.isnan(x_f32))
