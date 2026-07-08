# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
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

_WHERE_OPSET9_DTYPES = [
    (np.float16, TensorProto.FLOAT16, [[1.0], [2.0]], [[10.0, 20.0, 30.0]]),
    (np.float32, TensorProto.FLOAT, [[1.0], [2.0]], [[10.0, 20.0, 30.0]]),
    (np.float64, TensorProto.DOUBLE, [[1.0], [2.0]], [[10.0, 20.0, 30.0]]),
    (np.int32, TensorProto.INT32, [[1], [2]], [[10, 20, 30]]),
    (np.int64, TensorProto.INT64, [[1], [2]], [[10, 20, 30]]),
    (np.uint8, TensorProto.UINT8, [[1], [2]], [[10, 20, 30]]),
]

_FIXED_DTYPES_NO_BFLOAT16 = [
    (np.uint8, TensorProto.UINT8),
    (np.uint16, TensorProto.UINT16),
    (np.uint32, TensorProto.UINT32),
    (np.uint64, TensorProto.UINT64),
    (np.int8, TensorProto.INT8),
    (np.int16, TensorProto.INT16),
    (np.int32, TensorProto.INT32),
    (np.int64, TensorProto.INT64),
    (np.float16, TensorProto.FLOAT16),
    (np.float32, TensorProto.FLOAT),
    (np.float64, TensorProto.DOUBLE),
    (np.bool_, TensorProto.BOOL),
]


def _values(dtype):
    if dtype == np.bool_:
        return (
            np.array([[True], [False]], dtype=np.bool_),
            np.array([[False, True, False]], dtype=np.bool_),
        )
    if np.issubdtype(dtype, np.unsignedinteger):
        return (
            np.array([[1], [2]], dtype=dtype),
            np.array([[10, 20, 30]], dtype=dtype),
        )
    if np.issubdtype(dtype, np.integer):
        return (
            np.array([[-1], [2]], dtype=dtype),
            np.array([[10, -20, 30]], dtype=dtype),
        )
    return (
        np.array([[1.0], [2.0]], dtype=dtype),
        np.array([[10.0, 20.0, 30.0]], dtype=dtype),
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
    _WHERE_OPSET9_DTYPES,
)
def test_where_opset14_registered_dtypes(
    np_dtype, tensor_type, values_a, values_b
):
    cond = np.array([[True, False, True]], dtype=np.bool_)
    a = np.array(values_a, dtype=np_dtype)
    b = np.array(values_b, dtype=np_dtype)
    run_and_compare(
        "Where",
        inputs={"condition": cond, "X": a, "Y": b},
        outputs=[("Z", tensor_type)],
        opset=14,
    )


@pytest.mark.parametrize(("np_dtype", "tensor_type"), _FIXED_DTYPES_NO_BFLOAT16)
def test_where_opset16_fixed_size_dtypes(np_dtype, tensor_type):
    cond = np.array([[True, False, True]], dtype=np.bool_)
    a, b = _values(np_dtype)
    expected = np.where(cond, a, b)
    model = build_model(
        "Where",
        inputs={"condition": cond, "X": a, "Y": b},
        outputs=[("Z", tensor_type)],
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
