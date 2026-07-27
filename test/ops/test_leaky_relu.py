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
"""End-to-end CPU-vs-MUSA test for the LeakyRelu operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run,
    run_and_compare,
    run_with_iobinding,
)


def test_leaky_relu_default_alpha():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("LeakyRelu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_leaky_relu_custom_alpha():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "LeakyRelu",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.2},
    )


def test_leaky_relu_opset13_custom_alpha():
    x = np.random.default_rng(2).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "LeakyRelu",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.2},
        opset=13,
    )


def test_leaky_relu_opset6_float():
    x = np.random.default_rng(3).standard_normal((4, 8)).astype(np.float32)
    run_and_compare(
        "LeakyRelu",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.2},
        opset=6,
    )


@pytest.mark.parametrize(
    ("dtype", "tensor_type"),
    [
        (np.float16, TensorProto.FLOAT16),
        (np.float64, TensorProto.DOUBLE),
    ],
)
def test_leaky_relu_opset13_float_like_dtypes(dtype, tensor_type):
    x = np.random.default_rng(4).standard_normal((4, 8)).astype(dtype)
    attrs = {"alpha": 0.2}
    if dtype == np.float64:
        expected = np.where(x >= 0.0, x, attrs["alpha"] * x)
        model = build_model(
            "LeakyRelu",
            inputs={"X": x},
            outputs=[("Y", tensor_type)],
            attrs=attrs,
            opset=13,
        )
        (actual,) = run(model, {"X": x}, use_musa=True)
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
        return

    run_and_compare(
        "LeakyRelu",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        attrs=attrs,
        opset=13,
        rtol=2e-2,
        atol=2e-2,
    )


def test_leaky_relu_opset16_bfloat16():
    x_f32 = np.linspace(-2.0, 2.0, num=8, dtype=np.float32).reshape(2, 4)
    x = float32_to_bfloat16_bits(x_f32)
    x_bf16_f32 = bfloat16_bits_to_float32(x)
    expected = np.where(x_bf16_f32 >= 0.0, x_bf16_f32, 0.2 * x_bf16_f32)
    model = build_model_with_input_types(
        "LeakyRelu",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"alpha": 0.2},
        opset=16,
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_leaky_relu_scalar_input():
    x = np.array(-4.0, dtype=np.float32)
    run_and_compare(
        "LeakyRelu",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.25},
    )
