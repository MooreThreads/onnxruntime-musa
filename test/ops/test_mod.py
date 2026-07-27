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
"""End-to-end CPU-vs-MUSA tests for Mod."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_mod_int64_scalar_broadcast():
    x = np.arange(12, dtype=np.int64)
    y = np.array(4, dtype=np.int64)
    run_and_compare(
        "Mod",
        inputs={"A": x, "B": y},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"fmod": 0},
        opset=17,
        rtol=0,
        atol=0,
    )


def test_mod_int32_tensor_broadcast():
    x = np.arange(12, dtype=np.int32).reshape(3, 4)
    y = np.array([2, 3, 4, 5], dtype=np.int32)
    run_and_compare(
        "Mod",
        inputs={"A": x, "B": y},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"fmod": 0},
        opset=17,
        rtol=0,
        atol=0,
    )
