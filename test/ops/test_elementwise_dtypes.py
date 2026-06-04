# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Dtype coverage for MUSA elementwise kernels with CPU fallback disabled."""

import math

import numpy as np
import pytest

from op_test_utils import TensorProto, build_model, run, run_and_compare
from op_test_utils import (
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_with_iobinding,
)


def _values(dtype, shape=(2, 3)):
    if np.issubdtype(dtype, np.floating):
        return np.linspace(0.5, 3.0, num=np.prod(shape), dtype=np.float64).reshape(shape).astype(dtype)
    return np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)


def _bf16(values):
    return float32_to_bfloat16_bits(np.asarray(values, dtype=np.float32))


def _bf16_to_f32(values):
    return bfloat16_bits_to_float32(values)


def _rounded_bf16_f32(values):
    return _bf16_to_f32(_bf16(values))


def _run_musa_bfloat16_and_compare_expected(
    op_type,
    *,
    inputs,
    expected,
    input_types=None,
    output_type=TensorProto.BFLOAT16,
    attrs=None,
    rtol=2e-2,
    atol=2e-2,
):
    input_types = input_types or {name: TensorProto.BFLOAT16 for name in inputs}
    model = build_model_with_input_types(
        op_type,
        inputs=inputs,
        input_types=input_types,
        outputs=[("Y", output_type)],
        attrs=attrs,
        opset=19,
    )
    (actual,) = run_with_iobinding(
        model,
        inputs,
        input_types,
        [("Y", output_type, expected.shape)],
        use_musa=True,
    )
    if output_type == TensorProto.BOOL:
        np.testing.assert_array_equal(actual, expected)
        return

    np.testing.assert_allclose(
        _bf16_to_f32(actual),
        _rounded_bf16_f32(expected),
        rtol=rtol,
        atol=atol,
    )


def _run_musa_and_compare_expected(
    op_type,
    *,
    inputs,
    output_type,
    expected,
    attrs=None,
    rtol=1e-3,
    atol=1e-4,
):
    model = build_model(
        op_type,
        inputs=inputs,
        outputs=[("Y", output_type)],
        attrs=attrs,
        opset=19,
    )
    (actual,) = run(model, inputs, use_musa=True)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.uint8, TensorProto.UINT8),
        (np.uint16, TensorProto.UINT16),
        (np.uint32, TensorProto.UINT32),
        (np.uint64, TensorProto.UINT64),
        (np.int8, TensorProto.INT8),
        (np.int16, TensorProto.INT16),
        (np.int32, TensorProto.INT32),
        (np.int64, TensorProto.INT64),
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
@pytest.mark.parametrize("op_type", ["Add", "Sub", "Mul", "Div"])
def test_binary_numeric_opset14_dtypes(op_type, dtype, tensor_type):
    a = _values(dtype)
    b = np.ones((1, 3), dtype=dtype)
    if op_type == "Sub" and np.issubdtype(dtype, np.unsignedinteger):
        b = np.zeros((1, 3), dtype=dtype)
    if op_type == "Mul":
        b = np.full((1, 3), 2, dtype=dtype)
    run_and_compare(
        op_type,
        inputs={"A": a, "B": b},
        outputs=[("Y", tensor_type)],
        opset=19,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.int32, TensorProto.INT32),
        (np.int64, TensorProto.INT64),
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
def test_pow_dtypes(dtype, tensor_type):
    a = _values(dtype)
    b = np.full((1, 3), 2, dtype=dtype)
    run_and_compare(
        "Pow",
        inputs={"A": a, "B": b},
        outputs=[("Y", tensor_type)],
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    ("base_dtype", "base_type", "exp_dtype", "exp_type"),
    [
        (np.float16, TensorProto.FLOAT16, np.int32, TensorProto.INT32),
        (np.float32, TensorProto.FLOAT, np.int64, TensorProto.INT64),
        (np.int32, TensorProto.INT32, np.float32, TensorProto.FLOAT),
    ],
)
def test_pow_mixed_exponent_dtypes(base_dtype, base_type, exp_dtype, exp_type):
    a = _values(base_dtype)
    b = np.full((1, 3), 2, dtype=exp_dtype)
    run_and_compare(
        "Pow",
        inputs={"A": a, "B": b},
        outputs=[("Y", base_type)],
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("op_type", ["Add", "Sub", "Mul", "Div", "Pow"])
def test_bfloat16_binary_dtypes(op_type):
    a_f32 = np.linspace(0.5, 3.0, num=6, dtype=np.float32).reshape(2, 3)
    b_f32 = np.full((1, 3), 2.0 if op_type in {"Mul", "Pow"} else 1.0, dtype=np.float32)
    expected = {
        "Add": a_f32 + b_f32,
        "Sub": a_f32 - b_f32,
        "Mul": a_f32 * b_f32,
        "Div": a_f32 / b_f32,
        "Pow": np.power(a_f32, b_f32),
    }[op_type]
    _run_musa_bfloat16_and_compare_expected(
        op_type,
        inputs={"A": _bf16(a_f32), "B": _bf16(b_f32)},
        expected=expected,
    )


def test_bfloat16_pow_mixed_exponent_dtype():
    a_f32 = np.linspace(0.5, 3.0, num=6, dtype=np.float32).reshape(2, 3)
    b = np.array([[2, 3, 2]], dtype=np.int32)
    _run_musa_bfloat16_and_compare_expected(
        "Pow",
        inputs={"A": _bf16(a_f32), "B": b},
        input_types={"A": TensorProto.BFLOAT16, "B": TensorProto.INT32},
        expected=np.power(a_f32, b.astype(np.float32)),
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.uint32, TensorProto.UINT32),
        (np.uint64, TensorProto.UINT64),
        (np.int32, TensorProto.INT32),
        (np.int64, TensorProto.INT64),
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
@pytest.mark.parametrize("op_type", ["Max", "Min"])
def test_variadic_min_max_dtypes(op_type, dtype, tensor_type):
    a = _values(dtype)
    b = np.flip(a, axis=1).copy()
    c = np.ones((1, 3), dtype=dtype)
    run_and_compare(
        op_type,
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", tensor_type)],
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
def test_sum_float_like_dtypes(dtype, tensor_type):
    a = _values(dtype)
    b = np.ones((1, 3), dtype=dtype)
    c = np.full((2, 1), 2, dtype=dtype)
    run_and_compare(
        "Sum",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", tensor_type)],
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("op_type", ["Max", "Min"])
def test_bfloat16_variadic_min_max_dtypes(op_type):
    a_f32 = np.linspace(0.5, 3.0, num=6, dtype=np.float32).reshape(2, 3)
    b_f32 = np.flip(a_f32, axis=1).copy()
    c_f32 = np.ones((1, 3), dtype=np.float32)
    expected = np.maximum(np.maximum(a_f32, b_f32), c_f32)
    if op_type == "Min":
        expected = np.minimum(np.minimum(a_f32, b_f32), c_f32)
    _run_musa_bfloat16_and_compare_expected(
        op_type,
        inputs={"A": _bf16(a_f32), "B": _bf16(b_f32), "C": _bf16(c_f32)},
        expected=expected,
    )


def test_bfloat16_sum_dtype():
    a_f32 = np.linspace(0.5, 3.0, num=6, dtype=np.float32).reshape(2, 3)
    b_f32 = np.ones((1, 3), dtype=np.float32)
    c_f32 = np.full((2, 1), 2.0, dtype=np.float32)
    _run_musa_bfloat16_and_compare_expected(
        "Sum",
        inputs={"A": _bf16(a_f32), "B": _bf16(b_f32), "C": _bf16(c_f32)},
        expected=a_f32 + b_f32 + c_f32,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.uint8, TensorProto.UINT8),
        (np.uint16, TensorProto.UINT16),
        (np.uint32, TensorProto.UINT32),
        (np.uint64, TensorProto.UINT64),
        (np.int8, TensorProto.INT8),
        (np.int16, TensorProto.INT16),
        (np.int32, TensorProto.INT32),
        (np.int64, TensorProto.INT64),
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
def test_abs_dtypes(dtype, tensor_type):
    a = _values(dtype)
    if np.issubdtype(dtype, np.signedinteger) or np.issubdtype(dtype, np.floating):
        a = a - a.max()
    run_and_compare(
        "Abs",
        inputs={"X": a},
        outputs=[("Y", tensor_type)],
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
@pytest.mark.parametrize(
    "op_type",
    ["Relu", "LeakyRelu", "Sigmoid", "Tanh", "Sqrt", "Reciprocal", "Log", "Erf"],
)
def test_unary_float_like_dtypes(op_type, dtype, tensor_type):
    a = _values(dtype)
    attrs = {"alpha": 0.2} if op_type == "LeakyRelu" else None
    if dtype == np.float64 and op_type in {"LeakyRelu", "Erf"}:
        if op_type == "LeakyRelu":
            expected = np.where(a >= 0.0, a, attrs["alpha"] * a)
        else:
            expected = np.vectorize(math.erf)(a).astype(dtype)
        _run_musa_and_compare_expected(
            op_type,
            inputs={"X": a},
            output_type=tensor_type,
            expected=expected,
            attrs=attrs,
            rtol=2e-2,
            atol=2e-2,
        )
        return
    run_and_compare(
        op_type,
        inputs={"X": a},
        outputs=[("Y", tensor_type)],
        attrs=attrs,
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    "op_type",
    ["Abs", "Erf", "LeakyRelu", "Relu", "Sigmoid", "Sqrt", "Tanh"],
)
def test_bfloat16_unary_dtypes(op_type):
    x_f32 = np.linspace(-1.5, 2.0, num=6, dtype=np.float32).reshape(2, 3)
    if op_type == "Sqrt":
        x_f32 = np.linspace(0.5, 3.0, num=6, dtype=np.float32).reshape(2, 3)
    attrs = {"alpha": 0.2} if op_type == "LeakyRelu" else None
    if op_type == "Abs":
        expected = np.abs(x_f32)
    elif op_type == "Erf":
        expected = np.vectorize(math.erf)(x_f32).astype(np.float32)
    elif op_type == "LeakyRelu":
        expected = np.where(x_f32 >= 0.0, x_f32, attrs["alpha"] * x_f32)
    elif op_type == "Relu":
        expected = np.maximum(x_f32, 0.0)
    elif op_type == "Sigmoid":
        expected = 1.0 / (1.0 + np.exp(-x_f32))
    elif op_type == "Sqrt":
        expected = np.sqrt(x_f32)
    else:
        expected = np.tanh(x_f32)
    _run_musa_bfloat16_and_compare_expected(
        op_type,
        inputs={"X": _bf16(x_f32)},
        expected=expected,
        attrs=attrs,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.uint32, TensorProto.UINT32),
        (np.uint64, TensorProto.UINT64),
        (np.int32, TensorProto.INT32),
        (np.int64, TensorProto.INT64),
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
@pytest.mark.parametrize("op_type", ["Equal", "Greater"])
def test_compare_dtypes(op_type, dtype, tensor_type):
    a = _values(dtype)
    b = np.flip(a, axis=1).copy()
    run_and_compare(
        op_type,
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=19,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("op_type", ["Equal", "Greater"])
def test_bfloat16_compare_dtypes(op_type):
    a_bits = _bf16(np.linspace(0.5, 3.0, num=6, dtype=np.float32).reshape(2, 3))
    b_bits = _bf16(np.array([[1.0, 1.0, 2.0]], dtype=np.float32))
    a_f32 = _bf16_to_f32(a_bits)
    b_f32 = _bf16_to_f32(b_bits)
    expected = (a_f32 == b_f32) if op_type == "Equal" else (a_f32 > b_f32)
    _run_musa_bfloat16_and_compare_expected(
        op_type,
        inputs={"A": a_bits, "B": b_bits},
        expected=expected,
        output_type=TensorProto.BOOL,
        rtol=0,
        atol=0,
    )


def test_equal_bool_dtype():
    a = np.array([[True, False, True]], dtype=np.bool_)
    b = np.array([[True], [False]], dtype=np.bool_)
    run_and_compare(
        "Equal",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=19,
        rtol=0,
        atol=0,
    )
