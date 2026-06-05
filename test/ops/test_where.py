# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Where operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run,
    run_and_compare,
    run_with_iobinding,
)


def test_where_float_broadcast_opset16():
    cond = np.array([[True, False, True]], dtype=np.bool_)
    a = np.array([[1.0], [2.0]], dtype=np.float32)
    b = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    run_and_compare(
        "Where",
        inputs={"condition": cond, "X": a, "Y": b},
        outputs=[("Z", TensorProto.FLOAT)],
        opset=16,
    )


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "values_a", "values_b"),
    [
        (np.int32, TensorProto.INT32, [[1], [2]], [[10, 20, 30]]),
        (np.int64, TensorProto.INT64, [[1], [2]], [[10, 20, 30]]),
        (np.float16, TensorProto.FLOAT16, [[1.0], [2.0]], [[10.0, 20.0, 30.0]]),
    ],
)
def test_where_byte_copy_dtypes(np_dtype, tensor_type, values_a, values_b):
    cond = np.array([[True, False, True]], dtype=np.bool_)
    a = np.array(values_a, dtype=np_dtype)
    b = np.array(values_b, dtype=np_dtype)
    run_and_compare(
        "Where",
        inputs={"condition": cond, "X": a, "Y": b},
        outputs=[("Z", tensor_type)],
        opset=16,
    )


def test_where_bool_musa_only():
    cond = np.array([[True, False, True]], dtype=np.bool_)
    a = np.array([[True], [False]], dtype=np.bool_)
    b = np.array([[False, True, False]], dtype=np.bool_)
    expected = np.where(cond, a, b)
    model = build_model(
        "Where",
        inputs={"condition": cond, "X": a, "Y": b},
        outputs=[("Z", TensorProto.BOOL)],
        opset=16,
    )
    (actual,) = run(model, {"condition": cond, "X": a, "Y": b}, use_musa=True)
    np.testing.assert_array_equal(actual, expected)


def test_where_bfloat16():
    cond = np.array([[True, False, True]], dtype=np.bool_)
    a = float32_to_bfloat16_bits(np.array([[1.0], [2.0]], dtype=np.float32))
    b = float32_to_bfloat16_bits(np.array([[10.0, 20.0, 30.0]], dtype=np.float32))
    expected = np.where(cond, a, b)
    model = build_model_with_input_types(
        "Where",
        inputs={"condition": cond, "X": a, "Y": b},
        input_types={"X": TensorProto.BFLOAT16, "Y": TensorProto.BFLOAT16},
        outputs=[("Z", TensorProto.BFLOAT16)],
        opset=16,
    )
    (actual,) = run_with_iobinding(
        model,
        {"condition": cond, "X": a, "Y": b},
        {"X": TensorProto.BFLOAT16, "Y": TensorProto.BFLOAT16},
        [("Z", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)
