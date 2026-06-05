# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the ConstantOfShape operator."""

import numpy as np
import pytest
from onnx import helper

from op_test_utils import (
    TensorProto,
    float32_to_bfloat16_bits,
    run_model_and_compare,
    run_with_iobinding,
)


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


@pytest.mark.parametrize(
    ("tensor_type", "value", "output_type"),
    [
        (TensorProto.BOOL, [True], TensorProto.BOOL),
        (TensorProto.INT32, [7], TensorProto.INT32),
        (TensorProto.INT64, [7], TensorProto.INT64),
        (TensorProto.FLOAT16, [np.float16(2.5)], TensorProto.FLOAT16),
    ],
)
def test_constant_of_shape_byte_copy_dtypes(tensor_type, value, output_type):
    fill = helper.make_tensor("value", tensor_type, [1], value)
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=fill)
    graph = helper.make_graph(
        [node],
        "constant_of_shape_dtype_initializer",
        [],
        [helper.make_tensor_value_info("Y", output_type, [2, 3])],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    run_model_and_compare(model.SerializeToString(), {})


def test_constant_of_shape_bfloat16():
    value = float32_to_bfloat16_bits(np.array([2.5], dtype=np.float32))
    fill = helper.make_tensor(
        "value", TensorProto.BFLOAT16, [1], value.tobytes(), raw=True
    )
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=fill)
    graph = helper.make_graph(
        [node],
        "constant_of_shape_bfloat16_initializer",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 3])],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    expected = np.full((2, 3), value[0], dtype=np.uint16)
    (actual,) = run_with_iobinding(
        model.SerializeToString(),
        {},
        {},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)
