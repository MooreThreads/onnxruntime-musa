# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Reshape operator."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run, run_and_compare, run_model_and_compare


def test_reshape_float():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    shape = np.array([32, 16], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_reshape_with_minus_one():
    x = np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32)
    shape = np.array([16, -1], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_reshape_int64():
    x = np.arange(4096, dtype=np.int64).reshape(16, 16, 16)
    shape = np.array([64, 64], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.INT64)],
    )


def test_reshape_zero_copies_input_dim():
    x = np.random.default_rng(2).standard_normal((2, 3, 4)).astype(np.float32)
    shape = np.array([0, -1], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_reshape_bool_to_vector():
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    shape = np.array([4], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.BOOL)],
    )


def test_reshape_int32_with_allowzero():
    x = np.arange(0, dtype=np.int32).reshape(0, 3)
    shape = np.array([0, 3], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"allowzero": 1},
    )


def test_reshape_float16():
    x = np.random.default_rng(3).standard_normal((2, 3, 4)).astype(np.float16)
    shape = np.array([4, 6], dtype=np.int64)
    run_and_compare(
        "Reshape",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT16)],
        rtol=2e-2,
        atol=2e-2,
    )


def test_reshape_constant_output_feeds_musa_add():
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["constant_data"],
            value=helper.make_tensor(
                "constant_data_value",
                TensorProto.FLOAT,
                [30 * 12],
                np.arange(30 * 12, dtype=np.float32),
            ),
        ),
        helper.make_node("Reshape", ["constant_data", "shape"], ["reshaped"]),
        helper.make_node("Add", ["reshaped", "bias"], ["Y"]),
    ]
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [30, 12])
    model = build_graph_model(
        nodes,
        inputs={"bias": np.array(1.25, dtype=np.float32)},
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[shape],
        opset=17,
        name="reshape_constant_to_add",
    )
    run_model_and_compare(model, {"bias": np.array(1.25, dtype=np.float32)})


def test_reshape_string_opset19_cpu_memory_control_path():
    x = np.array([["a", "b"]], dtype=object)
    shape = helper.make_tensor("shape", TensorProto.INT64, [1], [2])
    node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
    graph = helper.make_graph(
        [node],
        "reshape_string_control_path",
        [helper.make_tensor_value_info("X", TensorProto.STRING, [1, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.STRING, [2])],
        initializer=[shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = min(model.ir_version, 10)
    model_bytes = model.SerializeToString()

    (expected,) = run(model_bytes, {"X": x}, use_musa=False)
    (actual,) = run(model_bytes, {"X": x}, use_musa=True)
    np.testing.assert_array_equal(actual, expected)
