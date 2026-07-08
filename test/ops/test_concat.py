# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Concat operator."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run_and_compare, run_model_and_compare


def test_concat_axis0():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_concat_axis1():
    a = np.random.default_rng(2).standard_normal((32, 16)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_int64_negative_axis():
    a = np.arange(512, dtype=np.int64).reshape(16, 32)
    b = np.arange(512, 1024, dtype=np.int64).reshape(16, 32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": -1},
    )


def test_concat_int32_three_inputs_axis2():
    a = np.arange(2 * 3 * 1, dtype=np.int32).reshape(2, 3, 1)
    b = np.arange(100, 100 + 2 * 3 * 2, dtype=np.int32).reshape(2, 3, 2)
    c = np.arange(200, 200 + 2 * 3 * 3, dtype=np.int32).reshape(2, 3, 3)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axis": 2},
    )


def test_concat_bool_negative_axis():
    a = np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_)
    b = np.array([[[False, True], [True, False]], [[True, True], [False, False]]], dtype=np.bool_)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"axis": -1},
    )


def test_concat_float16_axis1():
    a = np.random.default_rng(4).standard_normal((4, 2)).astype(np.float16)
    b = np.random.default_rng(5).standard_normal((4, 3)).astype(np.float16)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axis": 1},
    )


def test_concat_uint8_axis0():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    b = np.arange(100, 108, dtype=np.uint8).reshape(2, 4)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.UINT8)],
        attrs={"axis": 0},
    )

def test_concat_many_small_inputs_axis1():
    inputs = {}
    rng = np.random.default_rng(6)
    for i in range(40):
        width = 1 if i % 2 == 0 else 3
        inputs[f"X{i}"] = rng.standard_normal((8, width)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs=inputs,
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_many_small_inputs_near_direct_kernel_limit_axis1():
    inputs = {}
    rng = np.random.default_rng(16)
    for i in range(237):
        width = 1 if i % 5 else 3
        inputs[f"X{i}"] = rng.standard_normal((2, width)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs=inputs,
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_small_int64_shape_metadata_mixed_inputs():
    nodes = [
        helper.make_node("Constant", [], ["prefix"], value=helper.make_tensor("prefix_value", TensorProto.INT64, [1], [1])),
        helper.make_node("Unsqueeze", ["dynamic_a", "axes"], ["dynamic_a_unsqueezed"]),
        helper.make_node("Unsqueeze", ["dynamic_b", "axes"], ["dynamic_b_unsqueezed"]),
        helper.make_node(
            "Concat",
            ["prefix", "dynamic_a_unsqueezed", "dynamic_b_unsqueezed"],
            ["Y"],
            axis=0,
        ),
    ]
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    model = build_graph_model(
        nodes,
        inputs={
            "dynamic_a": np.array(7, dtype=np.int64),
            "dynamic_b": np.array(11, dtype=np.int64),
        },
        outputs=[("Y", TensorProto.INT64)],
        initializers=[axes],
        opset=17,
        name="concat_shape_metadata_graph",
    )
    run_model_and_compare(model, {"dynamic_a": np.array(7, dtype=np.int64), "dynamic_b": np.array(11, dtype=np.int64)}, rtol=0, atol=0)
