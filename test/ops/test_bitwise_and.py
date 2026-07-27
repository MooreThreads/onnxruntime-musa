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
"""End-to-end CPU-vs-MUSA test for the BitwiseAnd operator."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


def test_bitwise_and_int32_opset18():
    a = np.array([[1, 3], [7, 8]], dtype=np.int32)
    b = np.array([[1, 2], [3, 4]], dtype=np.int32)
    run_and_compare(
        "BitwiseAnd",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.INT32)],
        opset=18,
    )


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type"),
    [
        (np.uint8, TensorProto.UINT8),
        (np.uint16, TensorProto.UINT16),
        (np.uint32, TensorProto.UINT32),
        (np.uint64, TensorProto.UINT64),
        (np.int8, TensorProto.INT8),
        (np.int16, TensorProto.INT16),
        (np.int32, TensorProto.INT32),
        (np.int64, TensorProto.INT64),
    ],
)
def test_bitwise_and_integer_dtypes(np_dtype, tensor_type):
    a = np.array([[1, 3], [7, 8]], dtype=np_dtype)
    b = np.array([[1, 2], [3, 4]], dtype=np_dtype)
    run_and_compare(
        "BitwiseAnd",
        inputs={"A": a, "B": b},
        outputs=[("Y", tensor_type)],
        opset=18,
    )
