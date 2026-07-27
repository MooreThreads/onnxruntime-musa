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
"""End-to-end CPU-vs-MUSA test for the Unsqueeze operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_unsqueeze_axis0():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    axes = np.array([0], dtype=np.int64)
    run_and_compare("Unsqueeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.FLOAT)])


def test_unsqueeze_multiple_axes():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    axes = np.array([0, 2], dtype=np.int64)
    run_and_compare("Unsqueeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.FLOAT)])


def test_unsqueeze_negative_axis_int32():
    x = np.arange(2 * 3, dtype=np.int32).reshape(2, 3)
    axes = np.array([-1], dtype=np.int64)
    run_and_compare("Unsqueeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.INT32)])


def test_unsqueeze_bool_unsorted_axes():
    x = np.array([True, False, True], dtype=np.bool_)
    axes = np.array([2, 0], dtype=np.int64)
    run_and_compare("Unsqueeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.BOOL)])


def test_unsqueeze_float16():
    x = np.random.default_rng(2).standard_normal((2, 3)).astype(np.float16)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "Unsqueeze",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", TensorProto.FLOAT16)],
        rtol=2e-2,
        atol=2e-2,
    )
