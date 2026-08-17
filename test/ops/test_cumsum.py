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
"""End-to-end CPU-vs-MUSA tests for CumSum."""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run,
    run_and_compare,
    run_with_iobinding,
)


def test_cumsum_int32_axis0_opset17():
    x = np.array([2, 3, 5, 7, 11], dtype=np.int32)
    axis = np.array([0], dtype=np.int64)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.INT32)],
        opset=17,
        rtol=0,
        atol=0,
    )


def test_cumsum_float_axis1_exclusive_reverse_opset17():
    x = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float32)
    axis = np.array(1, dtype=np.int32)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"exclusive": 1, "reverse": 1},
        opset=17,
        rtol=1e-5,
        atol=1e-6,
    )


def test_cumsum_int64_negative_axis_opset17():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    axis = np.array([-1], dtype=np.int64)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.INT64)],
        opset=17,
        rtol=0,
        atol=0,
    )


def test_cumsum_bfloat16_opset19():
    x_f32 = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    axis = np.array(1, dtype=np.int64)
    model = build_model_with_input_types(
        "CumSum",
        inputs={"X": x, "axis": axis},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        opset=19,
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x, "axis": axis},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), np.cumsum(x_f32, axis=1), atol=0.05
    )


def test_cumsum_float16_opset19():
    x = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float16)
    axis = np.array([1], dtype=np.int64)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.FLOAT16)],
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


def test_cumsum_float64_opset19():
    x = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float64)
    axis = np.array([1], dtype=np.int64)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.DOUBLE)],
        opset=19,
        rtol=1e-12,
        atol=1e-12,
    )


def test_cumsum_uint32_opset13():
    x = np.array([[1, 2, 4], [3, 5, 7]], dtype=np.uint32)
    axis = np.array([1], dtype=np.int64)
    model = build_model("CumSum", {"X": x, "axis": axis},
                        [("Y", TensorProto.UINT32)], opset=13)
    (actual,) = run(model, {"X": x, "axis": axis}, use_musa=True)
    np.testing.assert_array_equal(actual, np.cumsum(x, axis=1, dtype=np.uint32))


def test_cumsum_uint64_opset13():
    x = np.array([[1, 2, 4], [3, 5, 7]], dtype=np.uint64)
    axis = np.array([1], dtype=np.int64)
    model = build_model("CumSum", {"X": x, "axis": axis},
                        [("Y", TensorProto.UINT64)], opset=13)
    (actual,) = run(model, {"X": x, "axis": axis}, use_musa=True)
    np.testing.assert_array_equal(actual, np.cumsum(x, axis=1, dtype=np.uint64))
