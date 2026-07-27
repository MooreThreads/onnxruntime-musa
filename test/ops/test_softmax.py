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
"""End-to-end CPU-vs-MUSA test for the Softmax operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_softmax_last_axis():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "Softmax",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": -1},
    )


def test_softmax_axis0():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "Softmax",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_softmax_3d_middle_axis():
    x = np.random.default_rng(2).standard_normal((2, 5, 4)).astype(np.float32)
    run_and_compare(
        "Softmax",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_softmax_vector_default_axis():
    x = np.random.default_rng(3).standard_normal((32,)).astype(np.float32)
    run_and_compare("Softmax", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_softmax_float16():
    x = np.random.default_rng(4).standard_normal((8, 16)).astype(np.float16)
    run_and_compare(
        "Softmax",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axis": -1},
        rtol=2e-2,
        atol=2e-2,
    )


def test_softmax_double():
    x = np.random.default_rng(5).standard_normal((8, 16)).astype(np.float64)
    run_and_compare(
        "Softmax",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        attrs={"axis": -1},
        rtol=1e-6,
        atol=1e-7,
    )


def test_softmax_bfloat16():
    x_f32 = np.random.default_rng(6).standard_normal((4, 8)).astype(np.float32)
    inputs = {"X": float32_to_bfloat16_bits(x_f32)}
    x_bf16_f32 = bfloat16_bits_to_float32(inputs["X"])
    shifted = x_bf16_f32 - np.max(x_bf16_f32, axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    input_types = {"X": TensorProto.BFLOAT16}
    model = build_model_with_input_types(
        "Softmax",
        inputs=inputs,
        input_types=input_types,
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"axis": -1},
    )
    (actual,) = run_with_iobinding(
        model,
        inputs,
        input_types,
        [("Y", TensorProto.BFLOAT16, x_f32.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )
