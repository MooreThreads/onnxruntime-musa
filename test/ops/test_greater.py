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
"""End-to-end CPU-vs-MUSA test for the Greater operator."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run_and_compare, run_model_and_compare


def test_greater_float_broadcast():
    a = np.random.default_rng(0).standard_normal((16, 1)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((1, 32)).astype(np.float32)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_greater_int64():
    a = np.arange(32, dtype=np.int64).reshape(8, 4)
    b = np.arange(31, -1, -1, dtype=np.int64).reshape(8, 4)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_greater_int64_constant_scalar_metadata():
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["zero"],
            value=helper.make_tensor("zero_value", TensorProto.INT64, [], [0]),
        ),
        helper.make_node("Greater", ["A", "zero"], ["Y"]),
    ]
    model = build_graph_model(
        nodes,
        inputs={"A": np.array([3], dtype=np.int64)},
        outputs=[("Y", TensorProto.BOOL)],
        opset=17,
        name="greater_int64_constant_scalar_metadata",
    )
    run_model_and_compare(model, {"A": np.array([3], dtype=np.int64)}, rtol=0, atol=0)


def test_greater_int32_scalar_broadcast():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array(3, dtype=np.int32)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_greater_float_multidirectional_broadcast():
    a = np.array([0.0, 2.0, 4.0], dtype=np.float32).reshape(1, 3, 1)
    b = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32).reshape(1, 1, 4)
    run_and_compare("Greater", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])
