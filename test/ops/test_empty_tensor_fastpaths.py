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
"""Empty tensor coverage for MUSA kernels that prefer muDNN fast paths."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


def test_transpose_empty_float_dim():
    x = np.empty((400, 0, 56), dtype=np.float32)
    run_and_compare(
        "Transpose",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"perm": [0, 2, 1]},
    )


@pytest.mark.parametrize(
    "op_type",
    [
        "Abs",
        "Ceil",
        "Erf",
        "Exp",
        "LeakyRelu",
        "Log",
        "Reciprocal",
        "Relu",
        "Round",
        "Sigmoid",
        "Sqrt",
        "Tanh",
    ],
)
def test_unary_empty_float_dim(op_type):
    x = np.empty((12, 0, 7), dtype=np.float32)
    attrs = {"alpha": 0.2} if op_type == "LeakyRelu" else None
    run_and_compare(
        op_type,
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs=attrs,
    )


def test_add_empty_broadcast():
    a = np.empty((400, 1, 0), dtype=np.float32)
    b = np.array(1.25, dtype=np.float32)
    run_and_compare("Add", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_sub_empty_broadcast():
    a = np.array(1.25, dtype=np.float32)
    b = np.empty((50, 1, 0), dtype=np.float32)
    run_and_compare("Sub", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_mul_empty_broadcast():
    a = np.empty((50, 1, 0), dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    run_and_compare("Mul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_div_empty_broadcast():
    a = np.empty((400, 1, 0), dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    run_and_compare("Div", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_min_empty_broadcast():
    a = np.empty((4, 0, 8), dtype=np.float32)
    b = np.empty((1, 0, 8), dtype=np.float32)
    run_and_compare("Min", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_max_empty_broadcast():
    a = np.empty((4, 0, 8), dtype=np.float32)
    b = np.empty((1, 0, 8), dtype=np.float32)
    run_and_compare("Max", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_sum_empty_broadcast():
    a = np.empty((4, 0, 8), dtype=np.float32)
    b = np.empty((1, 0, 8), dtype=np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
