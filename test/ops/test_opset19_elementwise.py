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
"""Opset 19 coverage for MUSA elementwise kernels."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_opset19_float_binary_elementwise():
    a = np.random.default_rng(0).uniform(0.5, 2.0, (2, 1, 3)).astype(np.float32)
    b = np.random.default_rng(1).uniform(0.5, 2.0, (1, 4, 3)).astype(np.float32)
    for op_type in ("Add", "Sub", "Mul", "Div", "Pow", "Max", "Min"):
        run_and_compare(
            op_type,
            inputs={"A": a, "B": b},
            outputs=[("Y", TensorProto.FLOAT)],
            opset=19,
        )


def test_opset19_pow_mixed_exponent_dtype():
    a = np.random.default_rng(11).uniform(0.5, 2.0, (2, 1, 3)).astype(np.float16)
    b = np.full((1, 4, 3), 2, dtype=np.int32)
    run_and_compare(
        "Pow",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT16)],
        opset=19,
        rtol=2e-2,
        atol=2e-2,
    )


def test_opset19_float_unary_elementwise():
    x = np.random.default_rng(2).uniform(0.5, 2.0, (2, 3, 4)).astype(np.float32)
    for op_type in ("Abs", "Erf", "Log", "Reciprocal", "Relu", "Sigmoid", "Sqrt", "Tanh"):
        run_and_compare(
            op_type,
            inputs={"X": x},
            outputs=[("Y", TensorProto.FLOAT)],
            opset=19,
        )


def test_opset19_leaky_relu():
    x = np.random.default_rng(3).standard_normal((2, 3, 4)).astype(np.float32)
    run_and_compare(
        "LeakyRelu",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.2},
        opset=19,
    )


def test_opset19_sum():
    a = np.random.default_rng(4).standard_normal((2, 3, 4)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((1, 3, 4)).astype(np.float32)
    c = np.random.default_rng(6).standard_normal((2, 1, 4)).astype(np.float32)
    run_and_compare(
        "Sum",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=19,
    )


def test_opset19_compare_and_bool():
    a = np.random.default_rng(7).standard_normal((2, 1, 4)).astype(np.float32)
    b = np.random.default_rng(8).standard_normal((1, 3, 4)).astype(np.float32)
    run_and_compare(
        "Equal",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=19,
    )
    run_and_compare(
        "Greater",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=19,
    )

    lhs = np.array([[True], [False]], dtype=np.bool_)
    rhs = np.array([[False, True, False]], dtype=np.bool_)
    run_and_compare(
        "Or",
        inputs={"A": lhs, "B": rhs},
        outputs=[("Y", TensorProto.BOOL)],
        opset=19,
    )
    run_and_compare(
        "Not",
        inputs={"X": lhs},
        outputs=[("Y", TensorProto.BOOL)],
        opset=19,
    )
