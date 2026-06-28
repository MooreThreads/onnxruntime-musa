# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Cast operator."""

import numpy as np
from onnx import helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_graph_model,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_model_and_compare,
    run_model_and_compare_with_cpu_fallback,
    run_and_compare,
    run_with_iobinding,
)


def test_cast_float_to_int64():
    x = np.random.default_rng(0).uniform(-5.0, 5.0, (16, 32)).astype(np.float32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"to": TensorProto.INT64},
    )


def test_cast_float_identity_large_tensor():
    x = np.random.default_rng(7).uniform(-5.0, 5.0, (30, 2083)).astype(np.float32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"to": TensorProto.FLOAT},
    )


def test_cast_int64_to_float():
    x = np.arange(-256, 256, dtype=np.int64).reshape(16, 32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"to": TensorProto.FLOAT},
    )


def test_cast_float_to_int32():
    x = np.random.default_rng(1).uniform(-5.0, 5.0, (16, 32)).astype(np.float32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"to": TensorProto.INT32},
    )


def test_cast_int32_to_bool():
    x = np.array([[0, 1, -2], [3, 0, 4]], dtype=np.int32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"to": TensorProto.BOOL},
    )


def test_cast_bool_to_int64():
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"to": TensorProto.INT64},
    )


def test_cast_int64_to_int32():
    x = np.arange(-12, 12, dtype=np.int64).reshape(4, 6)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"to": TensorProto.INT32},
    )


def test_cast_float16_to_float():
    x = np.random.default_rng(2).standard_normal((4, 8)).astype(np.float16)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"to": TensorProto.FLOAT},
    )


def test_cast_float_to_float16():
    x = np.random.default_rng(3).uniform(-5.0, 5.0, (4, 8)).astype(np.float32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"to": TensorProto.FLOAT16},
    )


def test_cast_double_to_float():
    x = np.random.default_rng(4).uniform(-5.0, 5.0, (4, 8)).astype(np.float64)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"to": TensorProto.FLOAT},
    )


def test_cast_uint8_to_int16():
    x = np.arange(24, dtype=np.uint8).reshape(4, 6)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT16)],
        attrs={"to": TensorProto.INT16},
    )


def test_cast_int16_to_uint32():
    x = np.arange(24, dtype=np.int16).reshape(4, 6)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.UINT32)],
        attrs={"to": TensorProto.UINT32},
    )


def test_cast_bfloat16_to_float():
    x_f32 = np.random.default_rng(5).uniform(-3.0, 3.0, (4, 8)).astype(np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    model = build_model_with_input_types(
        "Cast",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"to": TensorProto.FLOAT},
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.FLOAT, x.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(actual, bfloat16_bits_to_float32(x), rtol=0, atol=0)


def test_cast_float_to_bfloat16():
    x = np.random.default_rng(6).uniform(-3.0, 3.0, (4, 8)).astype(np.float32)
    expected = float32_to_bfloat16_bits(x)
    model = build_model_with_input_types(
        "Cast",
        inputs={"X": x},
        input_types={"X": TensorProto.FLOAT},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"to": TensorProto.BFLOAT16},
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.FLOAT},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)


def test_cast_shape_metadata_int64_to_int32():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node(
            "Cast", ["shape_i64"], ["shape_i32"], to=TensorProto.INT32
        ),
    ]
    model = build_graph_model(
        nodes,
        inputs={"X": x},
        outputs=[("shape_i32", TensorProto.INT32)],
        opset=17,
        name="cast_shape_metadata_graph",
    )
    (actual,) = run_model_and_compare_with_cpu_fallback(model, {"X": x})
    np.testing.assert_array_equal(actual, np.array([2, 3, 4], dtype=np.int32))


def test_cast_shape_metadata_int64_to_int64_identity():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node(
            "Cast", ["shape_i64"], ["shape_i64_copy"], to=TensorProto.INT64
        ),
    ]
    model = build_graph_model(
        nodes,
        inputs={"X": x},
        outputs=[("shape_i64_copy", TensorProto.INT64)],
        opset=17,
        name="cast_shape_metadata_identity_graph",
    )
    (actual,) = run_model_and_compare_with_cpu_fallback(model, {"X": x})
    np.testing.assert_array_equal(actual, np.array([2, 3, 4], dtype=np.int64))


def test_cast_identity_after_constant_of_shape():
    fill = helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [30, 2083])
    nodes = [
        helper.make_node("ConstantOfShape", ["shape"], ["zeros"], value=fill),
        helper.make_node("Cast", ["zeros"], ["Y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "cast_identity_after_constant_of_shape",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [30, 2083])],
        initializer=[shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    (actual,) = run_model_and_compare(model.SerializeToString(), {})
    np.testing.assert_array_equal(actual, np.zeros((30, 2083), dtype=np.float32))
