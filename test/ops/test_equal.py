# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Equal operator."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, run_model_and_compare, run_and_compare


def test_equal_float_broadcast():
    a = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    b = np.array([[0.0, 2.0, 2.0]], dtype=np.float32)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_equal_bool():
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    b = np.array([[True, True], [False, False]], dtype=np.bool_)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_equal_int32_scalar_broadcast():
    a = np.array([[1, 2, 1], [3, 1, 4]], dtype=np.int32)
    b = np.array(1, dtype=np.int32)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_equal_int64_multidirectional_broadcast():
    a = np.array([1, 2, 3], dtype=np.int64).reshape(1, 3, 1)
    b = np.array([1, 0, 3, 4], dtype=np.int64).reshape(1, 1, 4)
    run_and_compare("Equal", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])


def test_equal_string_opset19_feeds_cast_control_path():
    x = np.array([["target"]], dtype=object)
    shape = helper.make_tensor("shape", TensorProto.INT64, [1], [1])
    expected = helper.make_tensor("expected", TensorProto.STRING, [1], ["target"])
    nodes = [
        helper.make_node("Reshape", ["X", "shape"], ["reshaped"]),
        helper.make_node("Equal", ["reshaped", "expected"], ["eq"]),
        helper.make_node("Cast", ["eq"], ["Y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "equal_string_cast_control_path",
        [helper.make_tensor_value_info("X", TensorProto.STRING, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        initializer=[shape, expected],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = min(model.ir_version, 10)
    run_model_and_compare(model.SerializeToString(), {"X": x})


def test_equal_string_opset19_scalar_broadcast():
    x = np.array([["target"], ["other"], ["target"], [""]], dtype=object)
    expected = helper.make_tensor("expected", TensorProto.STRING, [1], ["target"])
    nodes = [helper.make_node("Equal", ["X", "expected"], ["Y"])]
    graph = helper.make_graph(
        nodes,
        "equal_string_scalar_broadcast",
        [helper.make_tensor_value_info("X", TensorProto.STRING, ["N", 1])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, None)],
        initializer=[expected],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = min(model.ir_version, 10)
    run_model_and_compare(model.SerializeToString(), {"X": x})
