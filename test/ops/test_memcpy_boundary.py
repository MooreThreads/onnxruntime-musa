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
"""CPU/device boundary copies (MemcpyFromHost / MemcpyToHost).

ORT inserts a `MemcpyFromHost` node when a device (MUSA) kernel consumes a
CPU-produced tensor, and a `MemcpyToHost` node when a CPU kernel consumes a
MUSA-produced tensor. These tests pin the behaviour that the plugin EP provides
the kernels for those nodes, so the copy runs on the plugin's own stream and
the `MemcpyFromHost` destination is a MUSA device tensor.

The consumer operators below (`Flatten`, `Transpose`) reject CPU memory on the
device path, so the tests are RED (device-tensor guard failure) unless the
boundary copy really lands in device memory.
"""

import numpy as np
import pytest
from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    run_model_and_compare_with_cpu_fallback,
)


def _cpu_produced_input_model(x, op_type, attrs, name):
    """`Sin` (CPU EP) -> `op_type` (MUSA EP): one inserted MemcpyFromHost."""
    return build_graph_model(
        [
            helper.make_node("Sin", ["X"], ["cpu_output"]),
            helper.make_node(op_type, ["cpu_output"], ["Y"], **attrs),
        ],
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        name=name,
    )


def test_memcpy_from_host_feeds_flatten_device_memory():
    x = np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(2, 3, 4)
    model = _cpu_produced_input_model(
        x, "Flatten", {"axis": 1}, "memcpy_from_host_flatten_graph"
    )
    run_model_and_compare_with_cpu_fallback(
        model, {"X": x}, rtol=0, atol=0, disable_ep_error_fallback=True
    )


def test_memcpy_from_host_runs_on_musa_kernel(monkeypatch):
    x = np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(2, 3, 4)
    model = _cpu_produced_input_model(
        x, "Flatten", {"axis": 1}, "memcpy_from_host_kernel_route_graph"
    )

    # Provenance guard: the boundary copy must execute through the plugin's own
    # MemcpyFromHost kernel, not the CPU EP fallback.
    monkeypatch.setenv("ORT_MUSA_TEST_FAIL_MEMCPY_KERNEL", "1")
    with pytest.raises(Exception, match="test-only MUSA memcpy kernel route"):
        run_model_and_compare_with_cpu_fallback(
            model, {"X": x}, disable_ep_error_fallback=True
        )


def test_memcpy_from_host_shared_by_multiple_musa_kernels():
    x = np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(2, 3, 4)
    model = build_graph_model(
        [
            helper.make_node("Sin", ["X"], ["cpu_output"]),
            helper.make_node("Flatten", ["cpu_output"], ["flat"], axis=1),
            helper.make_node("Transpose", ["cpu_output"], ["transposed"], perm=[2, 0, 1]),
        ],
        inputs={"X": x},
        outputs=[("flat", TensorProto.FLOAT), ("transposed", TensorProto.FLOAT)],
        name="memcpy_from_host_multiple_consumers_graph",
    )
    run_model_and_compare_with_cpu_fallback(
        model, {"X": x}, rtol=0, atol=0, disable_ep_error_fallback=True
    )


def test_memcpy_to_host_before_cpu_kernel(monkeypatch):
    x = np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 3, 4)
    model = build_graph_model(
        [
            helper.make_node("Relu", ["X"], ["relu_output"]),
            helper.make_node("Sin", ["relu_output"], ["Y"]),
        ],
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        name="memcpy_to_host_graph",
    )

    run_model_and_compare_with_cpu_fallback(
        model, {"X": x}, rtol=1e-6, atol=1e-7, disable_ep_error_fallback=True
    )

    # Provenance guard for the MUSA -> CPU direction as well.
    monkeypatch.setenv("ORT_MUSA_TEST_FAIL_MEMCPY_KERNEL", "1")
    with pytest.raises(Exception, match="test-only MUSA memcpy kernel route"):
        run_model_and_compare_with_cpu_fallback(
            model, {"X": x}, disable_ep_error_fallback=True
        )


def _if_branch():
    return helper.make_graph(
        [
            helper.make_node("Sin", ["X"], ["cpu_output"]),
            helper.make_node("Flatten", ["cpu_output"], ["branch_y"], axis=1),
        ],
        "branch",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, None)],
    )


def test_memcpy_from_host_inside_if_subgraph():
    x = np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(2, 3, 4)
    node = helper.make_node(
        "If", ["cond"], ["Y"], then_branch=_if_branch(), else_branch=_if_branch()
    )
    graph = helper.make_graph(
        [node],
        "memcpy_from_host_if_graph",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 10)
    run_model_and_compare_with_cpu_fallback(
        model.SerializeToString(),
        {"cond": np.array([True], dtype=np.bool_), "X": x},
        rtol=0,
        atol=0,
        disable_ep_error_fallback=True,
    )
