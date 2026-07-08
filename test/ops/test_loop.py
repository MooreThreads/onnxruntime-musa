# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for Loop."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, run_model_and_compare


def _build_loop_scan_model():
    one = helper.make_tensor("one", TensorProto.FLOAT, [2], [1.0, 1.0])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Add", ["x_in", "one"], ["x_out"]),
            helper.make_node("Identity", ["x_out"], ["scan_out"]),
        ],
        "loop_body",
        [
            helper.make_tensor_value_info("iter_num", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x_in", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x_out", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [2]),
        ],
        initializer=[one],
    )
    node = helper.make_node("Loop", ["M", "cond", "X"], ["Y", "Y_scan"], body=body)
    graph = helper.make_graph(
        [node],
        "loop_scan",
        [
            helper.make_tensor_value_info("M", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("Y_scan", TensorProto.FLOAT, None),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_loop_scan_output_opset13():
    model = _build_loop_scan_model()
    feeds = {
        "M": np.array(3, dtype=np.int64),
        "cond": np.array(True, dtype=np.bool_),
        "X": np.array([1.0, 2.0], dtype=np.float32),
    }
    run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-6)
