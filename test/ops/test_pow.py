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
"""End-to-end CPU-vs-MUSA test for the Pow operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_pow_float():
    base = np.random.default_rng(0).uniform(0.1, 3.0, (16, 32)).astype(np.float32)
    exp = np.random.default_rng(1).uniform(-2.0, 2.0, (16, 32)).astype(np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])


def test_pow_broadcast():
    base = np.random.default_rng(2).uniform(0.5, 2.0, (16, 32, 16)).astype(np.float32)
    exp = np.array([2.0], dtype=np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])


def test_pow_integer_exponents():
    base = np.array([[2.0, 3.0, 4.0], [0.5, 1.5, 2.5]], dtype=np.float32)
    exp = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])


def test_pow_bfloat16_scalar_exponent():
    base = float32_to_bfloat16_bits(
        np.random.default_rng(3).uniform(0.25, 3.0, (4, 8)).astype(np.float32)
    )
    exponent = np.asarray(
        float32_to_bfloat16_bits(np.array(2.0, dtype=np.float32)),
        dtype=np.uint16,
    )
    expected = np.power(
        bfloat16_bits_to_float32(base), bfloat16_bits_to_float32(exponent)
    )
    model = build_model_with_input_types(
        "Pow",
        inputs={"X": base, "Y": exponent},
        input_types={"X": TensorProto.BFLOAT16, "Y": TensorProto.BFLOAT16},
        outputs=[("Z", TensorProto.BFLOAT16)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": base, "Y": exponent},
        {"X": TensorProto.BFLOAT16, "Y": TensorProto.BFLOAT16},
        [("Z", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), expected, rtol=2e-2, atol=2e-2
    )
