# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end MUSA tests for RandomUniform operators."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model,
    run,
    run_with_iobinding,
    _make_session,
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
    with pytest.raises(Exception):
        run_with_iobinding(
            model,
            {},
            {},
            [("Y", TensorProto.BFLOAT16, (2, 3))],
            use_musa=True,
        )


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


@pytest.mark.parametrize("np_dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64, np.bool_,
])
def test_random_uniform_like_registered_integer_and_bool_carriers(np_dtype):
    x = np.zeros((2, 3), dtype=np_dtype)
    model = build_model(
        "RandomUniformLike", inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"dtype": TensorProto.FLOAT, "low": -1.0, "high": 2.0, "seed": 3.0}, opset=1,
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
    with pytest.raises(Exception):
        run_with_iobinding(
            model,
            {"X": x},
            {},
            [("Y", TensorProto.BFLOAT16, x.shape)],
            use_musa=True,
        )


def test_random_uniform_seed_reproduces_sequence_but_advances_each_run():
    model = build_model(
        "RandomUniform",
        inputs={},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"shape": [32], "dtype": TensorProto.FLOAT, "seed": 17.0},
        opset=1,
    )
    session_a = _make_session(model, use_musa=True)
    session_b = _make_session(model, use_musa=True)
    a0 = session_a.run(None, {})[0]
    a1 = session_a.run(None, {})[0]
    b0 = session_b.run(None, {})[0]
    b1 = session_b.run(None, {})[0]
    np.testing.assert_array_equal(a0, b0)
    np.testing.assert_array_equal(a1, b1)
    assert not np.array_equal(a0, a1)
