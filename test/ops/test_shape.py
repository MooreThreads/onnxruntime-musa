# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Shape operator."""

import numpy as np

from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    run_and_compare,
    run_model_and_compare,
)


def test_shape_float():
    x = np.random.default_rng(0).standard_normal((16, 32, 16)).astype(np.float32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_int64():
    x = np.arange(512, dtype=np.int64).reshape(16, 32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_bool_empty_dim():
    x = np.zeros((0, 3, 1), dtype=np.bool_)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_int32_scalar():
    x = np.array(7, dtype=np.int32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_uint8():
    x = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_device_output_feeds_cast_gather_indices():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    data = np.arange(10, dtype=np.float32)
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["indices"], to=TensorProto.INT32),
        helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0),
    ]
    model = build_graph_model(
        nodes,
        inputs={"X": x, "data": data},
        outputs=[("Y", TensorProto.FLOAT)],
        name="shape_cast_gather_indices",
    )
    run_model_and_compare(model, {"X": x, "data": data})
