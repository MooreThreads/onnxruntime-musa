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
        (TensorProto.UINT8, [7], TensorProto.UINT8),
        (TensorProto.UINT16, [7], TensorProto.UINT16),
        (TensorProto.UINT32, [7], TensorProto.UINT32),
        (TensorProto.UINT64, [7], TensorProto.UINT64),
        (TensorProto.INT8, [7], TensorProto.INT8),
        (TensorProto.INT16, [7], TensorProto.INT16),
        (TensorProto.INT32, [7], TensorProto.INT32),
        (TensorProto.INT64, [7], TensorProto.INT64),
        (TensorProto.FLOAT16, [np.float16(2.5)], TensorProto.FLOAT16),
        (TensorProto.FLOAT, [2.5], TensorProto.FLOAT),
        (TensorProto.DOUBLE, [2.5], TensorProto.DOUBLE),
        (TensorProto.BOOL, [True], TensorProto.BOOL),
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


def test_constant_of_shape_opset9_int32_value():
    fill = helper.make_tensor("value", TensorProto.INT32, [1], [7])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=fill)
    graph = helper.make_graph(
        [node],
        "constant_of_shape_opset9_int32",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [2, 3])],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    model.ir_version = min(model.ir_version, 10)
    run_model_and_compare(model.SerializeToString(), {})


def test_constant_of_shape_bfloat16_not_registered():
    value = float32_to_bfloat16_bits(np.array([2.5], dtype=np.float32))
    fill = helper.make_tensor(
        "value", TensorProto.BFLOAT16, [1], value.tobytes(), raw=True
    )
    node = helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=fill)
    graph = helper.make_graph(
        [node],
        "constant_of_shape_bfloat16_initializer",
        [helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    with pytest.raises(Exception):
        run_with_iobinding(
            model.SerializeToString(),
            {"shape": np.array([2, 3], dtype=np.int64)},
            {},
            [("Y", TensorProto.BFLOAT16, (2, 3))],
            use_musa=True,
        )


def test_constant_of_shape_opset9_bool_value():
    fill = helper.make_tensor("value", TensorProto.BOOL, [1], [True])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 2])
    node = helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=fill)
    graph = helper.make_graph(
        [node],
        "constant_of_shape_opset9_bool",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [2, 2])],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    model.ir_version = min(model.ir_version, 10)
    run_model_and_compare(model.SerializeToString(), {})
