# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ConstantOfShape operator."""

from onnx import helper

from op_test_utils import TensorProto, run_model_and_compare


def test_constant_of_shape_float_initializer_value():
    fill = helper.make_tensor("value", TensorProto.FLOAT, [1], [2.5])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=fill)
    graph = helper.make_graph(
        [node],
        "constant_of_shape_initializer",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    run_model_and_compare(model.SerializeToString(), {})
