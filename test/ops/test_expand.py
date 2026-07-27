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
"""End-to-end CPU-vs-MUSA test for the Expand operator."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run_and_compare, run_model_and_compare


def test_expand_float():
    x = np.random.default_rng(0).standard_normal((1, 3, 1)).astype(np.float32)
    shape = np.array([2, 3, 4], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.FLOAT)])


def test_expand_bool():
    x = np.array([[True], [False]], dtype=np.bool_)
    shape = np.array([2, 3], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.BOOL)])


def test_expand_int32_leading_dims():
    x = np.array([1, 2, 3], dtype=np.int32)
    shape = np.array([2, 1, 3], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.INT32)])


def test_expand_int64_middle_broadcast():
    x = np.arange(6, dtype=np.int64).reshape(1, 2, 3)
    shape = np.array([4, 2, 3], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.INT64)])


def test_expand_cpu_initializer_input():
    x = np.array([[1], [2]], dtype=np.int64)
    shape = np.array([2, 3], dtype=np.int64)
    node = helper.make_node("Expand", ["X", "shape"], ["Y"])
    initializer = helper.make_tensor("X", TensorProto.INT64, x.shape, x.reshape(-1))
    model = build_graph_model(
        [node],
        inputs={"shape": shape},
        outputs=[("Y", TensorProto.INT64)],
        initializers=[initializer],
        opset=17,
        name="expand_cpu_initializer_input",
    )
    run_model_and_compare(model, {"shape": shape}, rtol=0, atol=0)


def test_expand_float16():
    x = np.random.default_rng(4).standard_normal((1, 4)).astype(np.float16)
    shape = np.array([3, 4], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.FLOAT16)])


def test_expand_uint16_leading_dims():
    x = np.array([1, 2, 3], dtype=np.uint16)
    shape = np.array([2, 1, 3], dtype=np.int64)
    run_and_compare("Expand", inputs={"X": x, "shape": shape}, outputs=[("Y", TensorProto.UINT16)])


def test_expand_scalar_to_zero_dim_shape():
    x = np.array(1.0, dtype=np.float32)
    shape = np.array([4, 0], dtype=np.int64)
    run_and_compare(
        "Expand",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.FLOAT)],
    )
