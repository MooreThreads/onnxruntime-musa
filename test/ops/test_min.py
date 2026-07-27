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
"""End-to-end CPU-vs-MUSA test for the Min operator."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run_and_compare, run_model_and_compare


def test_min_float_binary_broadcast():
    a = np.random.default_rng(0).standard_normal((16, 1)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((1, 32)).astype(np.float32)
    run_and_compare("Min", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_min_float_variadic():
    a = np.random.default_rng(2).standard_normal((8, 4)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((8, 4)).astype(np.float32)
    c = np.random.default_rng(4).standard_normal((8, 4)).astype(np.float32)
    run_and_compare("Min", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])


def test_min_int32_variadic():
    a = np.arange(32, dtype=np.int32).reshape(8, 4)
    b = np.full((8, 4), 7, dtype=np.int32)
    c = np.arange(31, -1, -1, dtype=np.int32).reshape(8, 4)
    run_and_compare("Min", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.INT32)])


def test_min_int64_constant_metadata():
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["one"],
            value=helper.make_tensor("one_value", TensorProto.INT64, [1], [1]),
        ),
        helper.make_node("Min", ["one", "A"], ["Y"]),
    ]
    model = build_graph_model(
        nodes,
        inputs={"A": np.array([3], dtype=np.int64)},
        outputs=[("Y", TensorProto.INT64)],
        opset=17,
        name="min_int64_constant_metadata",
    )
    run_model_and_compare(model, {"A": np.array([3], dtype=np.int64)}, rtol=0, atol=0)


def test_min_int64_binary_broadcast():
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    b = np.array([2, -20, 4, -40], dtype=np.int64)
    run_and_compare("Min", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.INT64)])


def test_min_float_three_inputs_broadcast():
    a = np.random.default_rng(2).standard_normal((2, 3, 1)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1, 3, 4)).astype(np.float32)
    c = np.random.default_rng(4).standard_normal((2, 1, 4)).astype(np.float32)
    run_and_compare("Min", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)])
