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
"""End-to-end tests for the FusedGemm Plugin EP fusion."""

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, build_graph_model, musa_devices


def _run_fused_gemm_and_compare(model: bytes, feeds: dict[str, np.ndarray]):
    cpu_options = ort.SessionOptions()
    cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu_session = ort.InferenceSession(
        model,
        sess_options=cpu_options,
        providers=["CPUExecutionProvider"],
    )
    expected = cpu_session.run(None, feeds)

    devices = musa_devices()
    if not devices:
        raise RuntimeError("FusedGemm fusion test requires a MUSA device")
    musa_options = ort.SessionOptions()
    musa_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    musa_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    musa_options.add_provider_for_devices(devices, {})
    musa_session = ort.InferenceSession(model, sess_options=musa_options)
    actual = musa_session.run(None, feeds)

    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-3, atol=1e-3)


def test_matmul_add_tanh_fusion():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 4, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    bias = rng.standard_normal((12,)).astype(np.float32)

    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["M"]),
        helper.make_node("Add", ["M", "Bias"], ["MB"]),
        helper.make_node("Tanh", ["MB"], ["Y"]),
    ]
    feeds = {"A": a, "B": b, "Bias": bias}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="matmul_add_tanh_fusion_graph",
    )

    _run_fused_gemm_and_compare(model, feeds)


def test_matmul_add_tanh_fusion_with_initializer_inputs():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((2, 4, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    bias = rng.standard_normal((12,)).astype(np.float32)

    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["M"]),
        helper.make_node("Add", ["M", "Bias"], ["MB"]),
        helper.make_node("Tanh", ["MB"], ["Y"]),
    ]
    feeds = {"A": a}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(bias, "Bias"),
        ],
        name="matmul_add_tanh_initializer_fusion_graph",
    )

    _run_fused_gemm_and_compare(model, feeds)
