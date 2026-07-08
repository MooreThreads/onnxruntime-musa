# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Gather operator."""

import numpy as np
from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    run_and_compare,
    run_model_and_compare,
    run_model_and_compare_with_cpu_fallback,
)


def test_gather_axis0():
    data = np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32)
    indices = np.array([0, 8, 16, 24], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_gather_axis1():
    data = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    indices = np.array([[0, 8], [16, 24]], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_gather_int32_indices_negative_axis():
    data = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
    indices = np.array([[3, 1], [0, 2]], dtype=np.int32)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axis": -1},
    )


def test_gather_negative_indices():
    data = np.arange(3 * 4 * 2, dtype=np.int64).reshape(3, 4, 2)
    indices = np.array([-1, 0, -3], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": 1},
    )


def test_gather_bool_scalar_index():
    data = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    indices = np.array(1, dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"axis": 0},
    )


def test_gather_float16_axis1():
    data = np.random.default_rng(2).standard_normal((4, 8)).astype(np.float16)
    indices = np.array([1, 5, 7], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axis": 1},
    )


def test_gather_cpu_initializer_data_axis0():
    weights = np.random.default_rng(3).standard_normal((6, 4)).astype(np.float32)
    indices = np.array([0, 3, 5], dtype=np.int64)
    node = helper.make_node("Gather", ["weights", "indices"], ["Y"], axis=0)
    initializer = helper.make_tensor(
        "weights", TensorProto.FLOAT, weights.shape, weights.reshape(-1)
    )
    model = build_graph_model(
        [node],
        inputs={"indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[initializer],
        opset=17,
        name="gather_cpu_initializer_data_axis0",
    )
    run_model_and_compare(model, {"indices": indices})


def test_gather_uint16_int32_indices():
    data = np.arange(3 * 4, dtype=np.uint16).reshape(3, 4)
    indices = np.array([[0, 2], [3, 1]], dtype=np.int32)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.UINT16)],
        attrs={"axis": 1},
    )


def test_gather_shape_metadata_int64():
    x = np.zeros((2, 3, 4, 5), dtype=np.float32)
    indices = np.array([0, 2, 3], dtype=np.int64)
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node(
            "Gather", ["shape_i64", "indices"], ["selected_shape"], axis=0
        ),
    ]
    model = build_graph_model(
        nodes,
        inputs={"X": x, "indices": indices},
        outputs=[("selected_shape", TensorProto.INT64)],
        opset=17,
        name="gather_shape_metadata_graph",
    )
    (actual,) = run_model_and_compare_with_cpu_fallback(
        model, {"X": x, "indices": indices}
    )
    np.testing.assert_array_equal(actual, np.array([2, 4, 5], dtype=np.int64))
