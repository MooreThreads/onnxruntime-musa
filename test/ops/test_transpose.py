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
"""End-to-end CPU-vs-MUSA test for the Transpose operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_transpose_default():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Transpose", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_transpose_perm():
    x = np.random.default_rng(1).standard_normal((16, 32, 16)).astype(np.float32)
    run_and_compare(
        "Transpose",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"perm": [1, 2, 0]},
    )


def test_transpose_int64_4d():
    x = np.arange(2 * 3 * 4 * 5, dtype=np.int64).reshape(2, 3, 4, 5)
    run_and_compare(
        "Transpose",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"perm": [0, 2, 3, 1]},
    )


def test_transpose_bool_default_3d():
    x = np.array([[[True, False], [False, True], [True, True]]], dtype=np.bool_)
    run_and_compare("Transpose", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])


def test_transpose_float16_perm():
    x = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float16)
    run_and_compare(
        "Transpose",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"perm": [2, 0, 1]},
    )


def test_transpose_uint16_default():
    x = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    run_and_compare("Transpose", inputs={"X": x}, outputs=[("Y", TensorProto.UINT16)])


def test_transpose_bfloat16_perm():
    x_f32 = np.random.default_rng(3).standard_normal((2, 3, 4)).astype(np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    expected = np.transpose(x, (1, 2, 0))
    model = build_model_with_input_types(
        "Transpose",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"perm": [1, 2, 0]},
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)
