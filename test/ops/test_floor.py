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
"""End-to-end CPU-vs-MUSA tests for Floor."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_floor_float():
    x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32) * 4
    run_and_compare(
        "Floor",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=17,
        rtol=0,
        atol=0,
    )


def test_floor_scalar_double():
    x = np.array(-1.25, dtype=np.float64)
    run_and_compare(
        "Floor",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        opset=17,
        rtol=0,
        atol=0,
    )
