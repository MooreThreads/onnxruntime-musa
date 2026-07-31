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
"""End-to-end tests for the GemmActivation Plugin EP fusion."""

import numpy as np
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, build_graph_model, run_model_and_compare


def test_gemm_relu_fusion():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    c = rng.standard_normal((12,)).astype(np.float32)

    nodes = [
        helper.make_node("Gemm", ["A", "B", "C"], ["G"], alpha=1.0, beta=1.0),
        helper.make_node("Relu", ["G"], ["Y"]),
    ]
    feeds = {"A": a, "B": b, "C": c}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="gemm_relu_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_reshape_gemm_reshape_relu_unsqueeze_fusion():
    """Exercise the high-rank Gemm lowering emitted by ORT MatMulAddFusion."""
    rng = np.random.default_rng(12)
    a = rng.standard_normal((1, 2, 4)).astype(np.float32)
    b = rng.standard_normal((4, 3)).astype(np.float32)
    bias = rng.standard_normal((3,)).astype(np.float32)
    initializers = [
        numpy_helper.from_array(np.array([2, 4], dtype=np.int64), "in_shape"),
        numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "out_shape"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"),
    ]
    nodes = [
        helper.make_node("Reshape", ["A", "in_shape"], ["A_2d"]),
        helper.make_node("Gemm", ["A_2d", "B", "Bias"], ["G"]),
        helper.make_node("Reshape", ["G", "out_shape"], ["G_3d"]),
        helper.make_node("Relu", ["G_3d"], ["R"]),
        helper.make_node("Unsqueeze", ["R", "axes"], ["Y"]),
    ]
    feeds = {"A": a, "B": b, "Bias": bias}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        initializers=initializers,
        name="reshape_gemm_reshape_relu_unsqueeze_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_reshape_gemm_reshape_relu_reshape_fusion():
    rng = np.random.default_rng(13)
    a = rng.standard_normal((1, 2, 4)).astype(np.float32)
    b = rng.standard_normal((4, 3)).astype(np.float32)
    bias = rng.standard_normal((3,)).astype(np.float32)
    initializers = [
        numpy_helper.from_array(np.array([2, 4], dtype=np.int64), "in_shape"),
        numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "out_shape"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "tail_shape"),
    ]
    nodes = [
        helper.make_node("Reshape", ["A", "in_shape"], ["A_2d"]),
        helper.make_node("Gemm", ["A_2d", "B", "Bias"], ["G"]),
        helper.make_node("Reshape", ["G", "out_shape"], ["G_3d"]),
        helper.make_node("Relu", ["G_3d"], ["R"]),
        helper.make_node("Reshape", ["R", "tail_shape"], ["Y"]),
    ]
    feeds = {"A": a, "B": b, "Bias": bias}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        initializers=initializers,
        name="reshape_gemm_reshape_relu_reshape_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
