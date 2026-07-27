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
"""End-to-end CPU-vs-MUSA test for the If control-flow operator."""

import numpy as np
from onnx import helper

from op_test_utils import (
    TensorProto,
    float32_to_bfloat16_bits,
    run_model_and_compare,
    run_with_iobinding,
)


def _build_if_squeeze_identity_model():
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    then_graph = helper.make_graph(
        [helper.make_node("Squeeze", ["X", "axes"], ["then_y"])],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_y", TensorProto.FLOAT, None)],
        initializer=[axes],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["else_y"])],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_y", TensorProto.FLOAT, None)],
    )
    node = helper.make_node(
        "If",
        ["cond"],
        ["Y"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph(
        [node],
        "if_squeeze_identity",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_if_identity_model(tensor_type, opset):
    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["then_y"])],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_y", tensor_type, None)],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["else_y"])],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_y", tensor_type, None)],
    )
    node = helper.make_node(
        "If",
        ["cond"],
        ["Y"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph(
        [node],
        "if_identity",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("X", tensor_type, [2, 3]),
        ],
        [helper.make_tensor_value_info("Y", tensor_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_if_then_branch_squeeze():
    model = _build_if_squeeze_identity_model()
    feeds = {
        "cond": np.array([True], dtype=np.bool_),
        "X": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    }
    run_model_and_compare(model, feeds)


def test_if_opset13_int64_value():
    model = _build_if_identity_model(TensorProto.INT64, 13)
    feeds = {
        "cond": np.array([True], dtype=np.bool_),
        "X": np.arange(6, dtype=np.int64).reshape(2, 3),
    }
    run_model_and_compare(model, feeds)


def test_if_opset16_bfloat16_value():
    model = _build_if_identity_model(TensorProto.BFLOAT16, 16)
    x = float32_to_bfloat16_bits(
        np.linspace(-1.5, 1.5, 6, dtype=np.float32).reshape(2, 3)
    )
    feeds = {
        "cond": np.array([False], dtype=np.bool_),
        "X": x,
    }
    (actual,) = run_with_iobinding(
        model,
        feeds,
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, x)


def test_if_else_branch_identity():
    model = _build_if_squeeze_identity_model()
    feeds = {
        "cond": np.array([False], dtype=np.bool_),
        "X": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    }
    run_model_and_compare(model, feeds)
