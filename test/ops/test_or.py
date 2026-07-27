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
"""End-to-end CPU-vs-MUSA test for the Or operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_or_bool_broadcast():
    a = np.array([[True], [False]], dtype=np.bool_)
    b = np.array([[False, True, False]], dtype=np.bool_)
    run_and_compare("Or", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_or_bool_scalar_broadcast():
    a = np.array([[True, False], [False, False]], dtype=np.bool_)
    b = np.array(True, dtype=np.bool_)
    run_and_compare("Or", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_or_bool_multidirectional_broadcast():
    a = np.array([True, False, False], dtype=np.bool_).reshape(1, 3, 1)
    b = np.array([False, False, True, True], dtype=np.bool_).reshape(1, 1, 4)
    run_and_compare("Or", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])
