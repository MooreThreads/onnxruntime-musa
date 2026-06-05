# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Less operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_less_float_broadcast():
    a = np.array([[1.0], [3.0]], dtype=np.float32)
    b = np.array([[2.0, 3.0]], dtype=np.float32)
    run_and_compare("Less", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


@pytest.mark.parametrize(
    ("np_dtype", "values_a", "values_b"),
    [
        (np.int32, [[1], [3]], [[2, 3]]),
        (np.int64, [[1], [3]], [[2, 3]]),
        (np.uint32, [[1], [3]], [[2, 3]]),
        (np.uint64, [[1], [3]], [[2, 3]]),
        (np.float16, [[1.0], [3.0]], [[2.0, 3.0]]),
        (np.float64, [[1.0], [3.0]], [[2.0, 3.0]]),
    ],
)
def test_less_compare_dtypes(np_dtype, values_a, values_b):
    a = np.array(values_a, dtype=np_dtype)
    b = np.array(values_b, dtype=np_dtype)
    run_and_compare("Less", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_less_bfloat16():
    a_f32 = np.array([[1.0], [3.0]], dtype=np.float32)
    b_f32 = np.array([[2.0, 3.0]], dtype=np.float32)
    a = float32_to_bfloat16_bits(a_f32)
    b = float32_to_bfloat16_bits(b_f32)
    model = build_model_with_input_types(
        "Less",
        inputs={"A": a, "B": b},
        input_types={"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BOOL)],
    )
    expected = a_f32 < b_f32
    (actual,) = run_with_iobinding(
        model,
        {"A": a, "B": b},
        {"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        [("Y", TensorProto.BOOL, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)
