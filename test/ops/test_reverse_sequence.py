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
"""End-to-end CPU-vs-MUSA tests for ReverseSequence."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_reverse_sequence_time_major_float_opset10():
    x = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
    sequence_lens = np.array([5, 3, 1], dtype=np.int64)
    run_and_compare(
        "ReverseSequence",
        inputs={"X": x, "sequence_lens": sequence_lens},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"time_axis": 0, "batch_axis": 1},
        opset=10,
        rtol=1e-5,
        atol=1e-6,
    )


def test_reverse_sequence_batch_major_int64_opset10():
    x = np.arange(3 * 5 * 2, dtype=np.int64).reshape(3, 5, 2)
    sequence_lens = np.array([2, 5, 3], dtype=np.int64)
    run_and_compare(
        "ReverseSequence",
        inputs={"X": x, "sequence_lens": sequence_lens},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"time_axis": 1, "batch_axis": 0},
        opset=10,
        rtol=0,
        atol=0,
    )


def test_reverse_sequence_bool_opset10():
    x = np.array(
        [
            [[True], [False], [True], [False]],
            [[False], [True], [False], [True]],
        ],
        dtype=np.bool_,
    )
    sequence_lens = np.array([4, 2], dtype=np.int64)
    run_and_compare(
        "ReverseSequence",
        inputs={"X": x, "sequence_lens": sequence_lens},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"time_axis": 1, "batch_axis": 0},
        opset=10,
        rtol=0,
        atol=0,
    )
