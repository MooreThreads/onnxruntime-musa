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
"""End-to-end CPU-vs-MUSA test for the Mul operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_mul_float():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_mul_int64():
    a = np.arange(-256, 256, dtype=np.int64).reshape(16, 32)
    b = np.arange(1, 513, dtype=np.int64).reshape(16, 32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_mul_broadcast():
    a = np.random.default_rng(2).standard_normal((32, 16, 1)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 1, 32)).astype(np.float32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_mul_int32_scalar_broadcast():
    a = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)
    b = np.array(-3, dtype=np.int32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT32)])


def test_mul_int64_multidirectional_broadcast():
    a = np.arange(2 * 3 * 1, dtype=np.int64).reshape(2, 3, 1)
    b = np.array([-1, 2, -3, 4], dtype=np.int64).reshape(1, 1, 4)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_mul_bfloat16_broadcast():
    a = float32_to_bfloat16_bits(
        np.random.default_rng(6).standard_normal((2, 3, 4)).astype(np.float32)
    )
    b = float32_to_bfloat16_bits(
        np.random.default_rng(7).standard_normal((1, 3, 1)).astype(np.float32)
    )
    expected = bfloat16_bits_to_float32(a) * bfloat16_bits_to_float32(b)
    model = build_model_with_input_types(
        "Mul",
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
        bfloat16_bits_to_float32(actual), expected, rtol=1e-2, atol=1e-2
    )
