# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""All registered dtype coverage for low-coupling unary operators."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run,
    run_with_iobinding,
)


_FLOAT_DTYPES = [
    (np.float16, TensorProto.FLOAT16, 2e-2),
    (np.float32, TensorProto.FLOAT, 1e-5),
    (np.float64, TensorProto.DOUBLE, 1e-12),
]


@pytest.mark.parametrize("op_type", ["Abs", "Relu", "Sigmoid", "Tanh", "Neg", "Sign"])
@pytest.mark.parametrize("np_dtype,tensor_type,tol", _FLOAT_DTYPES)
def test_unary_float_dtypes(op_type, np_dtype, tensor_type, tol):
    x = np.array([[-8.0, -1.25, -0.0, 0.0, 1.25, 8.0]], dtype=np_dtype)
    run_and_compare(
        op_type,
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize("op_type", ["Abs", "Relu", "Sigmoid", "Tanh", "Neg", "Sign"])
def test_unary_bfloat16(op_type):
    x_f32 = np.array([[-8.0, -1.25, -0.0, 0.0, 1.25, 8.0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    reference_input = bfloat16_bits_to_float32(x)
    references = {
        "Abs": np.abs,
        "Relu": lambda value: np.maximum(value, 0.0),
        "Sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
        "Tanh": np.tanh,
        "Neg": np.negative,
        "Sign": np.sign,
    }
    expected_bits = float32_to_bfloat16_bits(references[op_type](reference_input))
    model = build_model_with_input_types(
        op_type,
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
    # Compare BF16 values rather than raw payloads: +0 and -0 are numerically
    # equivalent, and muDNN Relu preserves the input zero sign.
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        bfloat16_bits_to_float32(expected_bits),
        rtol=0,
        atol=0,
    )


_SIGNED_INTEGER_DTYPES = [
    (np.int8, TensorProto.INT8),
    (np.int16, TensorProto.INT16),
    (np.int32, TensorProto.INT32),
    (np.int64, TensorProto.INT64),
]
_UNSIGNED_INTEGER_DTYPES = [
    (np.uint8, TensorProto.UINT8),
    (np.uint16, TensorProto.UINT16),
    (np.uint32, TensorProto.UINT32),
    (np.uint64, TensorProto.UINT64),
]


@pytest.mark.parametrize("op_type", ["Abs", "Neg", "Sign"])
@pytest.mark.parametrize("np_dtype,tensor_type", _SIGNED_INTEGER_DTYPES)
def test_unary_signed_integer_dtypes(op_type, np_dtype, tensor_type):
    x = np.array([[-7, -1, 0, 1, 7]], dtype=np_dtype)
    run_and_compare(op_type, inputs={"X": x}, outputs=[("Y", tensor_type)], rtol=0, atol=0)


@pytest.mark.parametrize("op_type", ["Abs", "Sign"])
@pytest.mark.parametrize("np_dtype,tensor_type", _UNSIGNED_INTEGER_DTYPES)
def test_unary_unsigned_integer_dtypes(op_type, np_dtype, tensor_type):
    x = np.array([[0, 1, 7]], dtype=np_dtype)
    run_and_compare(op_type, inputs={"X": x}, outputs=[("Y", tensor_type)], rtol=0, atol=0)


@pytest.mark.parametrize(
    "op_type,np_dtype,tensor_type",
    [
        ("Abs", np.int32, TensorProto.INT32),
        ("Relu", np.float32, TensorProto.FLOAT),
        ("Sigmoid", np.float32, TensorProto.FLOAT),
        ("Tanh", np.float32, TensorProto.FLOAT),
        ("Neg", np.int32, TensorProto.INT32),
        ("Sign", np.uint32, TensorProto.UINT32),
    ],
)
def test_unary_empty_tensor(op_type, np_dtype, tensor_type):
    x = np.empty((2, 0, 3), dtype=np_dtype)
    run_and_compare(op_type, inputs={"X": x}, outputs=[("Y", tensor_type)], rtol=0, atol=0)

_FLOAT_UNARY_OPS = [
    "Ceil",
    "Erf",
    "Exp",
    "Floor",
    "Log",
    "Reciprocal",
    "LeakyRelu",
    "Softplus",
]


@pytest.mark.parametrize("op_type", _FLOAT_UNARY_OPS)
@pytest.mark.parametrize(
    "np_dtype,tensor_type,tol",
    [
        (np.float16, TensorProto.FLOAT16, 2e-2),
        (np.float32, TensorProto.FLOAT, 1e-5),
        (np.float64, TensorProto.DOUBLE, 1e-12),
    ],
)
def test_float_unary_registered_dtypes(op_type, np_dtype, tensor_type, tol):
    x = np.array([[0.25, 0.5, 1.0, 2.0]], dtype=np_dtype)
    attrs = {"alpha": 0.2} if op_type == "LeakyRelu" else None
    if np_dtype == np.float64 and op_type in {"Erf", "Softplus"}:
        model = build_model(
            op_type, inputs={"X": x}, outputs=[("Y", tensor_type)], attrs=attrs
        )
        (actual,) = run(model, {"X": x}, use_musa=True)
        if op_type == "Erf":
            expected = np.vectorize(lambda value: __import__("math").erf(value))(x)
        else:
            expected = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
        return
    run_and_compare(
        op_type,
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        attrs=attrs,
        rtol=tol,
        atol=tol,
    )


_BF16_REFERENCES = {
    "Ceil": np.ceil,
    "Erf": np.vectorize(lambda value: __import__("math").erf(value)),
    "Exp": np.exp,
    "Floor": np.floor,
    "Log": np.log,
    "Reciprocal": np.reciprocal,
    "LeakyRelu": lambda value: np.where(value >= 0.0, value, 0.2 * value),
}


@pytest.mark.parametrize("op_type", list(_BF16_REFERENCES))
def test_float_unary_bfloat16(op_type):
    x = float32_to_bfloat16_bits(
        np.array([[-2.0, -0.5, 0.25, 1.0, 2.0]], dtype=np.float32)
    )
    reference_input = bfloat16_bits_to_float32(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = _BF16_REFERENCES[op_type](reference_input).astype(np.float32)
    attrs = {"alpha": 0.2} if op_type == "LeakyRelu" else None
    model = build_model_with_input_types(
        op_type,
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs=attrs,
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
        bfloat16_bits_to_float32(float32_to_bfloat16_bits(expected)),
        rtol=2e-2,
        atol=2e-2,
        equal_nan=True,
    )


@pytest.mark.parametrize("op_type", _FLOAT_UNARY_OPS)
def test_float_unary_empty_tensor(op_type):
    x = np.empty((2, 0, 3), dtype=np.float32)
    attrs = {"alpha": 0.2} if op_type == "LeakyRelu" else None
    run_and_compare(
        op_type,
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs=attrs,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("op_type", ["Ceil", "Erf", "Exp", "Floor", "Log", "Reciprocal", "Softplus"])
def test_float_unary_nan_inf(op_type):
    x = np.array([np.nan, -np.inf, -1.0, -0.0, 0.0, 1.0, np.inf], dtype=np.float32)
    if op_type == "Erf":
        model = build_model(op_type, inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
        (actual,) = run(model, {"X": x}, use_musa=True)
        expected = np.array([np.nan, -1.0, -0.8427008, -0.0, 0.0, 0.8427008, 1.0], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)
        return
    run_and_compare(
        op_type,
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "np_dtype,tensor_type,values",
    [
        (np.int8, TensorProto.INT8, [-4, -1, 0, 3]),
        (np.uint8, TensorProto.UINT8, [0, 1, 2, 5]),
        (np.int16, TensorProto.INT16, [-4, -1, 0, 3]),
        (np.uint16, TensorProto.UINT16, [0, 1, 2, 5]),
        (np.int32, TensorProto.INT32, [-4, -1, 0, 3]),
        (np.uint32, TensorProto.UINT32, [0, 1, 2, 5]),
        (np.int64, TensorProto.INT64, [-4, -1, 0, 3]),
        (np.uint64, TensorProto.UINT64, [0, 1, 2, 5]),
    ],
)
def test_clip_all_integer_dtypes(np_dtype, tensor_type, values):
    x = np.array(values, dtype=np_dtype)
    lower = np.array(0 if np.issubdtype(np_dtype, np.unsignedinteger) else -1, dtype=np_dtype)
    upper = np.array(2, dtype=np_dtype)
    if np_dtype in {np.int16, np.uint16, np.int32, np.uint32}:
        model = build_model(
            "Clip",
            inputs={"X": x, "min": lower, "max": upper},
            outputs=[("Y", tensor_type)],
        )
        (actual,) = run(model, {"X": x, "min": lower, "max": upper}, use_musa=True)
        np.testing.assert_array_equal(actual, np.clip(x, lower, upper))
        return
    run_and_compare(
        "Clip",
        inputs={"X": x, "min": lower, "max": upper},
        outputs=[("Y", tensor_type)],
        rtol=0,
        atol=0,
    )


def test_clip_bfloat16_and_nan():
    x = float32_to_bfloat16_bits(np.array([np.nan, -2.0, 0.5, 2.0], dtype=np.float32))
    lower = float32_to_bfloat16_bits(np.array([-1.0], dtype=np.float32))
    upper = float32_to_bfloat16_bits(np.array([1.0], dtype=np.float32))
    model = build_model_with_input_types(
        "Clip",
        inputs={"X": x, "min": lower, "max": upper},
        input_types={
            "X": TensorProto.BFLOAT16,
            "min": TensorProto.BFLOAT16,
            "max": TensorProto.BFLOAT16,
        },
        outputs=[("Y", TensorProto.BFLOAT16)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x, "min": lower, "max": upper},
        {"X": TensorProto.BFLOAT16, "min": TensorProto.BFLOAT16, "max": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        np.array([np.nan, -1.0, 0.5, 1.0], dtype=np.float32),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "np_dtype,tensor_type",
    [
        (np.float16, TensorProto.FLOAT16),
        (np.float32, TensorProto.FLOAT),
        (np.float64, TensorProto.DOUBLE),
        (np.int32, TensorProto.INT32),
        (np.int64, TensorProto.INT64),
        (np.uint32, TensorProto.UINT32),
        (np.uint64, TensorProto.UINT64),
    ],
)
def test_prelu_broadcast_registered_dtypes(np_dtype, tensor_type):
    values = [[0, 1, 2], [1, 2, 3]] if np.issubdtype(np_dtype, np.unsignedinteger) else [[-4, -2, 0], [1, 2, 3]]
    x = np.array(values, dtype=np_dtype)
    slope = np.array([1, 2, 3], dtype=np_dtype)
    if np.issubdtype(np_dtype, np.unsignedinteger):
        model = build_model(
            "PRelu",
            inputs={"X": x, "slope": slope},
            outputs=[("Y", tensor_type)],
            opset=17,
        )
        (actual,) = run(model, {"X": x, "slope": slope}, use_musa=True)
        np.testing.assert_array_equal(actual, x)
        return
    run_and_compare(
        "PRelu",
        inputs={"X": x, "slope": slope},
        outputs=[("Y", tensor_type)],
        opset=17,
        rtol=2e-2 if np_dtype == np.float16 else 0,
        atol=2e-2 if np_dtype == np.float16 else 0,
    )


def test_prelu_bfloat16_broadcast():
    x = float32_to_bfloat16_bits(np.array([[-4.0, -2.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32))
    slope = float32_to_bfloat16_bits(np.array([0.25, 0.5, 0.75], dtype=np.float32))
    model = build_model_with_input_types(
        "PRelu",
        inputs={"X": x, "slope": slope},
        input_types={"X": TensorProto.BFLOAT16, "slope": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        opset=17,
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x, "slope": slope},
        {"X": TensorProto.BFLOAT16, "slope": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    expected = np.where(
        bfloat16_bits_to_float32(x) >= 0.0,
        bfloat16_bits_to_float32(x),
        bfloat16_bits_to_float32(x) * bfloat16_bits_to_float32(slope),
    )
    np.testing.assert_allclose(bfloat16_bits_to_float32(actual), expected, rtol=2e-2, atol=2e-2)
