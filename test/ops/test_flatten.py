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
"""End-to-end CPU-vs-MUSA test for the Flatten operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


_HFD_DTYPES = [
    (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
    (np.float32, TensorProto.FLOAT, 1e-5, 1e-6),
    (np.float64, TensorProto.DOUBLE, 1e-12, 1e-12),
]

_FIXED_DTYPES_NO_BFLOAT16 = [
    (np.uint8, TensorProto.UINT8),
    (np.uint16, TensorProto.UINT16),
    (np.uint32, TensorProto.UINT32),
    (np.uint64, TensorProto.UINT64),
    (np.int8, TensorProto.INT8),
    (np.int16, TensorProto.INT16),
    (np.int32, TensorProto.INT32),
    (np.int64, TensorProto.INT64),
    (np.float16, TensorProto.FLOAT16),
    (np.float32, TensorProto.FLOAT),
    (np.float64, TensorProto.DOUBLE),
    (np.bool_, TensorProto.BOOL),
]


def _values(dtype):
    if dtype == np.bool_:
        return (np.arange(24) % 2 == 0).reshape(2, 3, 4)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.arange(24, dtype=dtype).reshape(2, 3, 4)
    if np.issubdtype(dtype, np.integer):
        return (np.arange(24).reshape(2, 3, 4) - 12).astype(dtype)
    return np.linspace(-1.5, 1.5, 24, dtype=dtype).reshape(2, 3, 4)


@pytest.mark.parametrize(("np_dtype", "tensor_type", "rtol", "atol"), _HFD_DTYPES)
def test_flatten_opset1_hfd_dtypes(np_dtype, tensor_type, rtol, atol):
    x = _values(np_dtype)
    run_and_compare(
        "Flatten",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        attrs={"axis": 1},
        opset=1,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(("np_dtype", "tensor_type"), _FIXED_DTYPES_NO_BFLOAT16)
def test_flatten_opset9_fixed_size_dtypes(np_dtype, tensor_type):
    x = _values(np_dtype)
    run_and_compare(
        "Flatten",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        attrs={"axis": 1},
        opset=9,
    )


def test_flatten_negative_axis():
    x = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    run_and_compare(
        "Flatten",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": -1},
        opset=13,
    )


def test_flatten_bfloat16_opset13():
    x = float32_to_bfloat16_bits(
        np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(2, 3, 4)
    )
    expected_shape = (2, 12)
    model = build_model_with_input_types(
        "Flatten",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"axis": 1},
        opset=13,
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected_shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, x.reshape(expected_shape))
