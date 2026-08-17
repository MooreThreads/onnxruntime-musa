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
"""End-to-end CPU-vs-MUSA tests for ScatterElements."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare
from op_test_utils import build_model_with_input_types, float32_to_bfloat16_bits, run_with_iobinding


def test_scatter_elements_float_axis1_none():
    data = np.zeros((2, 4), dtype=np.float32)
    indices = np.array([[1, 3], [0, 2]], dtype=np.int64)
    updates = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    run_and_compare(
        "ScatterElements",
        inputs={"data": data, "indices": indices, "updates": updates},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
        opset=17,
        rtol=0,
        atol=0,
    )


def test_scatter_elements_int64_axis0_add():
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([0, 2, 2], dtype=np.int64)
    updates = np.array([10, 20, 30], dtype=np.int64)
    run_and_compare(
        "ScatterElements",
        inputs={"data": data, "indices": indices, "updates": updates},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": 0, "reduction": "add"},
        opset=17,
        rtol=0,
        atol=0,
    )

@pytest.mark.parametrize(("dtype", "tensor_type"), [
    (np.uint8, TensorProto.UINT8), (np.int8, TensorProto.INT8),
    (np.int16, TensorProto.INT16), (np.uint16, TensorProto.UINT16),
    (np.uint32, TensorProto.UINT32), (np.uint64, TensorProto.UINT64),
    (np.float16, TensorProto.FLOAT16), (np.float64, TensorProto.DOUBLE),
])
def test_scatter_elements_dtype_matrix(dtype, tensor_type):
    data = np.zeros((2, 4), dtype=dtype)
    indices = np.array([[1, 3], [0, 2]], dtype=np.int64)
    updates = np.array([[1, 2], [3, 4]], dtype=np.int64).astype(dtype)
    run_and_compare("ScatterElements", inputs={"data": data, "indices": indices, "updates": updates},
                    outputs=[("Y", tensor_type)], attrs={"axis": 1}, opset=17)


def test_scatter_elements_bool_none():
    data = np.zeros((2, 4), dtype=np.bool_)
    indices = np.array([[1, 3], [0, 2]], dtype=np.int64)
    updates = np.array([[True, False], [True, False]], dtype=np.bool_)
    run_and_compare("ScatterElements", inputs={"data": data, "indices": indices, "updates": updates},
                    outputs=[("Y", TensorProto.BOOL)], attrs={"axis": 1}, opset=17,
                    rtol=0, atol=0)


def test_scatter_elements_bfloat16_none():
    data = float32_to_bfloat16_bits(np.zeros((2, 4), dtype=np.float32))
    updates = float32_to_bfloat16_bits(np.array([[1, 2], [3, 4]], dtype=np.float32))
    indices = np.array([[1, 3], [0, 2]], dtype=np.int64)
    model = build_model_with_input_types("ScatterElements",
        inputs={"data": data, "indices": indices, "updates": updates},
        input_types={"data": TensorProto.BFLOAT16, "indices": TensorProto.INT64, "updates": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)], attrs={"axis": 1}, opset=17)
    expected = data.copy(); expected[0, [1, 3]] = updates[0]; expected[1, [0, 2]] = updates[1]
    (actual,) = run_with_iobinding(model, {"data": data, "indices": indices, "updates": updates},
        {"data": TensorProto.BFLOAT16, "indices": TensorProto.INT64, "updates": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)], use_musa=True)
    np.testing.assert_array_equal(actual, expected)
