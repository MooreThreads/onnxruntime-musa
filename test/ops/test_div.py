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
"""End-to-end CPU-vs-MUSA test for the Div operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_div_float():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).uniform(1.0, 2.0, (16, 32)).astype(np.float32)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_div_int64():
    # Exact divisors to avoid truncation-direction ambiguity between EPs.
    a = np.arange(2, 16 * 32 * 2 + 1, 2, dtype=np.int64).reshape(16, 32)
    b = np.full((16, 32), 2, dtype=np.int64)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_div_broadcast():
    a = np.random.default_rng(2).standard_normal((16, 32, 16)).astype(np.float32)
    b = np.random.default_rng(3).uniform(1.0, 2.0, (16,)).astype(np.float32)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_div_int32_scalar_broadcast():
    a = np.arange(12, 24, dtype=np.int32).reshape(3, 4)
    b = np.array(3, dtype=np.int32)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT32)])


def test_div_float_multidirectional_broadcast():
    a = np.random.default_rng(2).uniform(1.0, 5.0, (2, 3, 1)).astype(np.float32)
    b = np.random.default_rng(3).uniform(0.5, 2.0, (1, 1, 4)).astype(np.float32)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
