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
"""End-to-end CPU-vs-MUSA test for the Sqrt operator."""

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


def test_sqrt_float():
    x = np.random.default_rng(0).uniform(0.0, 10.0, (16, 32)).astype(np.float32)
    run_and_compare("Sqrt", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_sqrt_float_matrix_with_zero():
    x = np.array([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=np.float32)
    run_and_compare("Sqrt", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "tol"),
    [
        (np.float16, TensorProto.FLOAT16, 2e-2),
        (np.float64, TensorProto.DOUBLE, 1e-12),
    ],
)
def test_sqrt_float_like_dtypes(np_dtype, tensor_type, tol):
    x = np.array([[0.0, 0.25, 1.0], [2.0, 16.0, 100.0]], dtype=np_dtype)
    run_and_compare(
        "Sqrt",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        rtol=tol,
        atol=tol,
    )


def test_sqrt_bfloat16_zero_and_positive_values():
    x = float32_to_bfloat16_bits(
        np.array([[0.0, 0.25, 1.0], [2.0, 16.0, 100.0]], dtype=np.float32)
    )
    expected = np.sqrt(bfloat16_bits_to_float32(x))
    model = build_model_with_input_types(
        "Sqrt",
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
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), expected, rtol=1e-2, atol=1e-2
    )
