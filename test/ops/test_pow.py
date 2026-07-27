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
"""End-to-end CPU-vs-MUSA test for the Pow operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_pow_float():
    base = np.random.default_rng(0).uniform(0.1, 3.0, (16, 32)).astype(np.float32)
    exp = np.random.default_rng(1).uniform(-2.0, 2.0, (16, 32)).astype(np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])


def test_pow_broadcast():
    base = np.random.default_rng(2).uniform(0.5, 2.0, (16, 32, 16)).astype(np.float32)
    exp = np.array([2.0], dtype=np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])


def test_pow_integer_exponents():
    base = np.array([[2.0, 3.0, 4.0], [0.5, 1.5, 2.5]], dtype=np.float32)
    exp = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])
