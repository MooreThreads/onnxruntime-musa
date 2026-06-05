# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end MUSA tests for RandomUniform operators."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model,
    run,
    run_with_iobinding,
)


def test_random_uniform_opset1_runs_on_musa_without_cpu_fallback():
    model = build_model(
        "RandomUniform",
        inputs={},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "shape": [2, 3],
            "dtype": TensorProto.FLOAT,
            "low": -1.0,
            "high": 2.0,
            "seed": 3.0,
        },
        opset=1,
    )
    (actual,) = run(model, {}, use_musa=True)
    assert actual.shape == (2, 3)
    assert actual.dtype == np.float32
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 2.0)


@pytest.mark.parametrize(
    ("tensor_type", "np_dtype"),
    [
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.DOUBLE, np.float64),
    ],
)
def test_random_uniform_float_like_dtypes(tensor_type, np_dtype):
    model = build_model(
        "RandomUniform",
        inputs={},
        outputs=[("Y", tensor_type)],
        attrs={
            "shape": [2, 3],
            "dtype": tensor_type,
            "low": -1.0,
            "high": 2.0,
            "seed": 3.0,
        },
        opset=1,
    )
    (actual,) = run(model, {}, use_musa=True)
    assert actual.shape == (2, 3)
    assert actual.dtype == np_dtype
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 2.0)


def test_random_uniform_bfloat16():
    model = build_model(
        "RandomUniform",
        inputs={},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={
            "shape": [2, 3],
            "dtype": TensorProto.BFLOAT16,
            "low": -1.0,
            "high": 2.0,
            "seed": 3.0,
        },
        opset=1,
    )
    (actual,) = run_with_iobinding(
        model,
        {},
        {},
        [("Y", TensorProto.BFLOAT16, (2, 3))],
        use_musa=True,
    )
    actual_f32 = bfloat16_bits_to_float32(actual)
    assert actual.shape == (2, 3)
    assert np.all(actual_f32 >= -1.0)
    assert np.all(actual_f32 <= 2.0)


def test_random_uniform_like_float_opset1_runs_on_musa_without_cpu_fallback():
    x = np.zeros((2, 3), dtype=np.float32)
    model = build_model(
        "RandomUniformLike",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"low": -1.0, "high": 2.0, "seed": 3.0},
        opset=1,
    )
    (actual,) = run(model, {"X": x}, use_musa=True)
    assert actual.shape == x.shape
    assert actual.dtype == np.float32
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 2.0)


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type"),
    [
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
def test_random_uniform_like_float_like_input_dtypes(np_dtype, tensor_type):
    x = np.zeros((2, 3), dtype=np_dtype)
    model = build_model(
        "RandomUniformLike",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        attrs={"low": -1.0, "high": 2.0, "seed": 3.0},
        opset=1,
    )
    (actual,) = run(model, {"X": x}, use_musa=True)
    assert actual.shape == x.shape
    assert actual.dtype == np_dtype
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 2.0)


@pytest.mark.parametrize(
    ("tensor_type", "np_dtype"),
    [
        (TensorProto.FLOAT16, np.float16),
        (TensorProto.DOUBLE, np.float64),
    ],
)
def test_random_uniform_like_dtype_attr_path(tensor_type, np_dtype):
    x = np.zeros((2, 3), dtype=np.int32)
    model = build_model(
        "RandomUniformLike",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        attrs={"dtype": tensor_type, "low": -1.0, "high": 2.0, "seed": 3.0},
        opset=1,
    )
    (actual,) = run(model, {"X": x}, use_musa=True)
    assert actual.shape == x.shape
    assert actual.dtype == np_dtype
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 2.0)


def test_random_uniform_like_bfloat16_dtype_attr_path():
    x = np.zeros((2, 3), dtype=np.int32)
    model = build_model(
        "RandomUniformLike",
        inputs={"X": x},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={
            "dtype": TensorProto.BFLOAT16,
            "low": -1.0,
            "high": 2.0,
            "seed": 3.0,
        },
        opset=1,
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    actual_f32 = bfloat16_bits_to_float32(actual)
    assert actual.shape == x.shape
    assert np.all(actual_f32 >= -1.0)
    assert np.all(actual_f32 <= 2.0)
