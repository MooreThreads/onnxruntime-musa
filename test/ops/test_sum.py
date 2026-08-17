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
"""End-to-end CPU-vs-MUSA test for the Sum (variadic) operator."""

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


def test_sum_two_inputs_same_shape():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_three_inputs():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)
    c = rng.standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_many_inputs_same_shape():
    rng = np.random.default_rng(3)
    inputs = {
        f"X{i}": rng.standard_normal((8, 16)).astype(np.float32)
        for i in range(10)
    }
    run_and_compare("Sum", inputs=inputs, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_broadcast():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((32,)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_three_inputs_broadcast():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((16, 1)).astype(np.float32)
    b = rng.standard_normal((1, 32)).astype(np.float32)
    c = rng.standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_three_inputs_broadcast_no_output_shape_input():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((16, 1, 1)).astype(np.float32)
    b = rng.standard_normal((1, 32, 1)).astype(np.float32)
    c = rng.standard_normal((1, 1, 8)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_float_three_inputs_multidirectional_broadcast():
    a = np.random.default_rng(2).standard_normal((2, 3, 1)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 1, 4)).astype(np.float32)
    c = np.random.default_rng(4).standard_normal((1, 3, 1)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_float_scalar_broadcast():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array(100.0, dtype=np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
        (np.float64, TensorProto.DOUBLE, 1e-6, 1e-7),
    ],
)
def test_sum_other_float_schema_dtypes(np_dtype, tensor_type, rtol, atol):
    a = np.arange(12, dtype=np_dtype).reshape(3, 4)
    b = np.array([1, 2, 3, 4], dtype=np_dtype)
    run_and_compare(
        "Sum",
        inputs={"A": a, "B": b},
        outputs=[("Y", tensor_type)],
        rtol=rtol,
        atol=atol,
    )


def test_sum_bfloat16_broadcast():
    a_f32 = np.arange(12, dtype=np.float32).reshape(3, 4)
    b_f32 = np.array([1, 2, 3, 4], dtype=np.float32)
    a = float32_to_bfloat16_bits(a_f32)
    b = float32_to_bfloat16_bits(b_f32)
    expected = bfloat16_bits_to_float32(a) + bfloat16_bits_to_float32(b)
    model = build_model_with_input_types(
        "Sum",
        inputs={"A": a, "B": b},
        input_types={"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"A": a, "B": b},
        {"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), expected, rtol=2e-2, atol=2e-2
    )


def test_sum_one_input_device_copy():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    run_and_compare("Sum", inputs={"A": a}, outputs=[("Y", TensorProto.DOUBLE)])


def test_sum_empty_output():
    a = np.empty((2, 0, 3), dtype=np.float16)
    b = np.empty((1, 0, 1), dtype=np.float16)
    run_and_compare(
        "Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT16)]
    )
