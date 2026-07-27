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
"""End-to-end CPU-vs-MUSA test for the Range operator."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "start", "limit", "delta", "rtol", "atol"),
    [
        (np.int16, TensorProto.INT16, 1, 8, 2, 0, 0),
        (np.int32, TensorProto.INT32, 1, 8, 2, 0, 0),
        (np.int64, TensorProto.INT64, -3, 4, 3, 0, 0),
        (np.float32, TensorProto.FLOAT, 0.5, 2.0, 0.5, 1e-5, 1e-6),
        (np.float64, TensorProto.DOUBLE, 0.5, 2.0, 0.5, 1e-12, 1e-12),
    ],
)
def test_range_registered_dtypes(
    np_dtype, tensor_type, start, limit, delta, rtol, atol
):
    run_and_compare(
        "Range",
        inputs={
            "start": np.array(start, dtype=np_dtype),
            "limit": np.array(limit, dtype=np_dtype),
            "delta": np.array(delta, dtype=np_dtype),
        },
        outputs=[("Y", tensor_type)],
        opset=11,
        rtol=rtol,
        atol=atol,
    )


def test_range_double_descending_empty():
    run_and_compare(
        "Range",
        inputs={
            "start": np.array(5.0, dtype=np.float64),
            "limit": np.array(1.0, dtype=np.float64),
            "delta": np.array(1.0, dtype=np.float64),
        },
        outputs=[("Y", TensorProto.DOUBLE)],
        opset=11,
        rtol=1e-12,
        atol=1e-12,
    )
