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
"""End-to-end CPU-vs-MUSA test for the Relu operator."""

import numpy as np
from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    run_and_compare,
    run_model_and_compare_with_cpu_fallback,
)


def test_relu_float():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Relu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_relu_float_5d():
    x = np.random.default_rng(1).standard_normal((2, 3, 1, 4, 5)).astype(np.float32)
    run_and_compare("Relu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_relu_float_matrix_with_negatives():
    x = np.array([[-3.5, -0.0, 2.0], [0.0, -7.25, 9.5]], dtype=np.float32)
    run_and_compare("Relu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_relu_float16_accepts_cpu_produced_input(monkeypatch):
    x = np.array([[-3.5, -0.0, 0.0, 2.0, 9.5]], dtype=np.float16)
    model = build_graph_model(
        [
            helper.make_node("Sin", ["X"], ["cpu_output"]),
            helper.make_node("Relu", ["cpu_output"], ["Y"]),
        ],
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        name="relu_float16_cpu_produced_input_graph",
    )

    monkeypatch.setenv("ORT_MUSA_TEST_FAIL_RELU_CPU_INPUT", "1")
    # The plugin now runs the MemcpyFromHost node, so the CPU-produced tensor
    # reaches Relu in device memory and the test-only CPU-input route must
    # not be taken.
    run_model_and_compare_with_cpu_fallback(
        model, {"X": x}, disable_ep_error_fallback=True
    )

    monkeypatch.delenv("ORT_MUSA_TEST_FAIL_RELU_CPU_INPUT")
    run_model_and_compare_with_cpu_fallback(
        model,
        {"X": x},
        rtol=0,
        atol=0,
        disable_ep_error_fallback=True,
    )
