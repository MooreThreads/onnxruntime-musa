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
"""End-to-end CPU-vs-MUSA test for the Erf operator."""

import math

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


def test_erf_float():
    x = np.random.default_rng(0).uniform(-2.0, 2.0, (16, 32)).astype(np.float32)
    run_and_compare("Erf", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_erf_float_matrix():
    x = np.array([[-3.0, -1.0, 0.0], [0.5, 1.0, 3.0]], dtype=np.float32)
    run_and_compare("Erf", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "tensor_type,np_dtype,tol",
    [
        (TensorProto.FLOAT16, np.float16, 2e-3),
        (TensorProto.BFLOAT16, np.float32, 8e-3),
        (TensorProto.FLOAT, np.float32, 1e-6),
        (TensorProto.DOUBLE, np.float64, 1e-12),
    ],
    ids=["fp16", "bf16", "fp32", "fp64"],
)
def test_erf_special_values(tensor_type, np_dtype, tol):
    """muDNN can return SUCCESS while producing NaN for infinite inputs."""
    values = np.array(
        [-np.inf, -1e4, -3.0, -1.0, -0.0, 0.0, 0.5, 1.0, 3.0, 1e4, np.inf, np.nan],
        dtype=np_dtype,
    )
    is_bf16 = tensor_type == TensorProto.BFLOAT16
    x = float32_to_bfloat16_bits(values) if is_bf16 else values
    reference_input = bfloat16_bits_to_float32(x) if is_bf16 else x
    expected = np.array([math.erf(float(value)) for value in reference_input], dtype=np_dtype)
    if is_bf16:
        expected = bfloat16_bits_to_float32(float32_to_bfloat16_bits(expected))
    model = build_model_with_input_types(
        "Erf", inputs={"X": x}, input_types={"X": tensor_type}, outputs=[("Y", tensor_type)]
    )
    # The helper requires a MUSA device and disables CPU EP fallback, including
    # for BF16/DOUBLE where the CPU kernel is not our numerical reference.
    (output,) = run_with_iobinding(
        model, {"X": x}, {"X": tensor_type}, [("Y", tensor_type, x.shape)], use_musa=True
    )
    actual = bfloat16_bits_to_float32(output) if is_bf16 else output
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol, equal_nan=True)
    np.testing.assert_array_equal(actual[np.isinf(reference_input)], [-1.0, 1.0])
    np.testing.assert_array_equal(actual[reference_input == 0], 0.0)
    np.testing.assert_array_equal(
        np.signbit(actual[reference_input == 0]), np.signbit(reference_input[reference_input == 0])
    )
