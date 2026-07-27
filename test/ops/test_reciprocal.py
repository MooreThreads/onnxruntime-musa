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
"""End-to-end CPU-vs-MUSA test for the Reciprocal operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reciprocal_float():
    x = np.random.default_rng(0).uniform(0.5, 5.0, (16, 32)).astype(np.float32)
    run_and_compare("Reciprocal", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_reciprocal_float_matrix():
    x = np.random.default_rng(2).uniform(0.25, 4.0, (3, 4)).astype(np.float32)
    run_and_compare("Reciprocal", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
