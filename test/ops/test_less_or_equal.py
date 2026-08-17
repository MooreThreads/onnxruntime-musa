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
"""End-to-end CPU-vs-MUSA tests for LessOrEqual."""

import numpy as np
import pytest

from op_test_utils import TensorProto, build_model_with_input_types, float32_to_bfloat16_bits, run_and_compare, run_with_iobinding


def test_less_or_equal_int64_broadcast():
    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    b = np.array([0, 3, 4], dtype=np.int64)
    run_and_compare(
        "LessOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=17,
        rtol=0,
        atol=0,
    )


def test_less_or_equal_float_scalar():
    a = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)
    b = np.array(0.0, dtype=np.float32)
    run_and_compare(
        "LessOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=17,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32, np.uint32, np.uint64])
def test_less_or_equal_dtype_matrix(dtype):
    values = [2, 0, 3, 7] if np.issubdtype(dtype, np.unsignedinteger) else [-2, 0, 3, 7]
    a = np.array(values, dtype=dtype)
    b = np.array([0, 1, 2, 8], dtype=dtype)
    if dtype in (np.uint32, np.uint64):
        tensor_type = TensorProto.UINT32 if dtype == np.uint32 else TensorProto.UINT64
        model = build_model_with_input_types(
            "LessOrEqual", inputs={"A": a, "B": b},
            input_types={"A": tensor_type, "B": tensor_type},
            outputs=[("Y", TensorProto.BOOL)], opset=17)
        (actual,) = run_with_iobinding(
            model, {"A": a, "B": b}, {"A": tensor_type, "B": tensor_type},
            [("Y", TensorProto.BOOL, a.shape)], use_musa=True)
        np.testing.assert_array_equal(actual, a <= b)
    else:
        run_and_compare("LessOrEqual", inputs={"A": a, "B": b},
                        outputs=[("Y", TensorProto.BOOL)], opset=17,
                        rtol=0, atol=0)


def test_less_or_equal_bfloat16():
    a_f = np.array([-2.0, 0.0, 3.0, 7.0], dtype=np.float32)
    b_f = np.array([0.0, 1.0, 2.0, 8.0], dtype=np.float32)
    a = float32_to_bfloat16_bits(a_f)
    b = float32_to_bfloat16_bits(b_f)
    model = build_model_with_input_types(
        "LessOrEqual", inputs={"A": a, "B": b},
        input_types={"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BOOL)], opset=17)
    (actual,) = run_with_iobinding(model, {"A": a, "B": b},
        {"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        [("Y", TensorProto.BOOL, a.shape)], use_musa=True)
    np.testing.assert_array_equal(actual, a_f <= b_f)
