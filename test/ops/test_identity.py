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
"""End-to-end CPU-vs-MUSA test for the Identity operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


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
        return (np.arange(12) % 2 == 0).reshape(3, 4)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.arange(12, dtype=dtype).reshape(3, 4)
    if np.issubdtype(dtype, np.integer):
        return (np.arange(12).reshape(3, 4) - 6).astype(dtype)
    return np.linspace(-1.5, 1.5, 12, dtype=dtype).reshape(3, 4)


def test_identity_opset1_no_bfloat16_constraint():
    x = np.linspace(-1.5, 1.5, 12, dtype=np.float64).reshape(3, 4)
    run_and_compare(
        "Identity",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        opset=1,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(("np_dtype", "tensor_type"), _FIXED_DTYPES_NO_BFLOAT16)
def test_identity_opset19_fixed_size_dtypes(np_dtype, tensor_type):
    x = _values(np_dtype)
    run_and_compare(
        "Identity",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        opset=19,
    )


def test_identity_bfloat16_opset13():
    x = float32_to_bfloat16_bits(
        np.linspace(-1.5, 1.5, 12, dtype=np.float32).reshape(3, 4)
    )
    model = build_model_with_input_types(
        "Identity",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        opset=13,
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, x)
