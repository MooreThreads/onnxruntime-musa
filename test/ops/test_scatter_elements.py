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

from op_test_utils import TensorProto, run_and_compare


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
