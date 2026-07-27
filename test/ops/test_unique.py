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
"""End-to-end CPU-vs-MUSA tests for Unique."""

import numpy as np
from op_test_utils import TensorProto, build_model, run, run_and_compare


def _unique_reference(x, *, sorted_values):
    values = []
    first_indices = []
    counts = []
    for i, value in enumerate(x.tolist()):
        if value not in values:
            values.append(value)
            first_indices.append(i)
            counts.append(1)
        else:
            counts[values.index(value)] += 1

    if sorted_values:
        order = sorted(range(len(values)), key=lambda i: values[i])
        values = [values[i] for i in order]
        first_indices = [first_indices[i] for i in order]
        counts = [counts[i] for i in order]

    inverse = np.array([values.index(value) for value in x.tolist()], dtype=np.int64)
    return (
        np.array(values, dtype=x.dtype),
        np.array(first_indices, dtype=np.int64),
        inverse,
        np.array(counts, dtype=np.int64),
    )


def _run_musa_unique(x, tensor_type, attrs):
    model = build_model(
        "Unique",
        inputs={"X": x},
        outputs=[
            ("Y", tensor_type),
            ("indices", TensorProto.INT64),
            ("inverse_indices", TensorProto.INT64),
            ("counts", TensorProto.INT64),
        ],
        attrs=attrs,
        opset=17,
    )
    return run(model, {"X": x}, use_musa=True)


def test_unique_sorted0_flattened_int64():
    x = np.array([5, 2, 5, 3, 2, 9, 3, 3], dtype=np.int64)
    run_and_compare(
        "Unique",
        inputs={"X": x},
        outputs=[
            ("Y", TensorProto.INT64),
            ("indices", TensorProto.INT64),
            ("inverse_indices", TensorProto.INT64),
            ("counts", TensorProto.INT64),
        ],
        attrs={"sorted": 0},
        opset=17,
        rtol=0,
        atol=0,
    )



def test_unique_sorted1_axis0_1d_int64():
    x = np.array([4, 1, 4, 2, 9, 1, 2, 8, 9], dtype=np.int64)
    run_and_compare(
        "Unique",
        inputs={"X": x},
        outputs=[
            ("Y", TensorProto.INT64),
            ("indices", TensorProto.INT64),
            ("inverse_indices", TensorProto.INT64),
            ("counts", TensorProto.INT64),
        ],
        attrs={"axis": 0, "sorted": 1},
        opset=17,
        rtol=0,
        atol=0,
    )


def test_unique_int32_musa_against_numpy_reference():
    x = np.array([4, 1, 4, 2, 9, 1, 2, 8, 9], dtype=np.int32)
    expected = _unique_reference(x, sorted_values=True)
    actual = _run_musa_unique(x, TensorProto.INT32, {"axis": 0, "sorted": 1})
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


def test_unique_empty_int64():
    x = np.array([], dtype=np.int64)
    run_and_compare(
        "Unique",
        inputs={"X": x},
        outputs=[
            ("Y", TensorProto.INT64),
            ("indices", TensorProto.INT64),
            ("inverse_indices", TensorProto.INT64),
            ("counts", TensorProto.INT64),
        ],
        attrs={"sorted": 0},
        opset=17,
        rtol=0,
        atol=0,
    )
