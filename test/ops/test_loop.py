# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end CPU-vs-MUSA tests for Loop."""

import numpy as np
import pytest
from onnx import helper

from op_test_utils import TensorProto, run, run_model_and_compare


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


@pytest.mark.parametrize("np_dtype,tensor_type", [
    (np.float16, TensorProto.FLOAT16), (np.float64, TensorProto.DOUBLE),
    (np.int16, TensorProto.INT16), (np.int32, TensorProto.INT32),
    (np.int8, TensorProto.INT8), (np.uint8, TensorProto.UINT8),
    (np.uint16, TensorProto.UINT16), (np.uint32, TensorProto.UINT32),
    (np.uint64, TensorProto.UINT64),
    (np.float32, TensorProto.FLOAT),
])
def test_loop_registered_carried_dtypes(np_dtype, tensor_type):
    # Keep the body contract identical while varying the carried/scan dtype.
    body = helper.make_graph(
        [helper.make_node("Identity", ["cond_in"], ["cond_out"]),
         helper.make_node("Identity", ["x_in"], ["x_out"]),
         helper.make_node("Identity", ["x_out"], ["scan_out"])],
        "loop_body", [helper.make_tensor_value_info("iter_num", TensorProto.INT64, []),
                      helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                      helper.make_tensor_value_info("x_in", tensor_type, [2])],
        [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
         helper.make_tensor_value_info("x_out", tensor_type, [2]),
         helper.make_tensor_value_info("scan_out", tensor_type, [2])])
    node = helper.make_node("Loop", ["M", "cond", "X"], ["Y", "Y_scan"], body=body)
    graph = helper.make_graph(
        [node], "loop_dtype", [helper.make_tensor_value_info("M", TensorProto.INT64, []),
        helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        helper.make_tensor_value_info("X", tensor_type, [2])],
        [helper.make_tensor_value_info("Y", tensor_type, [2]),
         helper.make_tensor_value_info("Y_scan", tensor_type, None)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    x = np.array([1, 2], dtype=np_dtype)
    actual, scan = run(model.SerializeToString(), {"M": np.array(2, dtype=np.int64),
        "cond": np.array(True), "X": x}, use_musa=True)
    np.testing.assert_array_equal(actual, x)
    np.testing.assert_array_equal(scan, np.stack([x, x]))
