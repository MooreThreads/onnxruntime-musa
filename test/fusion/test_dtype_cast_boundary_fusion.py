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
"""Tests for preserving low-precision Cast boundaries around fusion."""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime as ort
import numpy as np
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, float32_to_bfloat16_bits, musa_devices


def _profile_node_names(
    model: bytes,
    feeds: dict[str, np.ndarray],
    tmp_path,
    name: str,
    *,
    outputs: list[tuple[str, int, tuple[int, ...]]] | None = None,
) -> list[str]:
    devices = musa_devices()
    if not devices:
        raise RuntimeError("Cast boundary fusion test requires a MUSA device")
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / name)
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model, sess_options=so)
    if outputs is None:
        session.run(None, feeds)
    else:
        io_binding = session.io_binding()
        for input_name, value in feeds.items():
            elem_type = helper.np_dtype_to_tensor_dtype(value.dtype)
            io_binding.bind_input(
                input_name, "cpu", 0, elem_type, value.shape, value.ctypes.data
            )
        output_buffers = []
        for output_name, elem_type, shape in outputs:
            output = np.empty(shape, dtype=np.uint16)
            io_binding.bind_output(
                output_name, "cpu", 0, elem_type, output.shape, output.ctypes.data
            )
            output_buffers.append(output)
        session.run_with_iobinding(io_binding)
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


def _assert_cast_boundary_preserved(
    model: bytes,
    feeds: dict[str, np.ndarray],
    tmp_path,
    name: str,
    *,
    outputs: list[tuple[str, int, tuple[int, ...]]] | None = None,
) -> list[str]:
    node_names = _profile_node_names(model, feeds, tmp_path, name, outputs=outputs)
    assert any(name.startswith("Cast_") for name in node_names)
    return node_names


def _assert_cast_boundary_not_fused(
    model: bytes, feeds: dict[str, np.ndarray], tmp_path, name: str
) -> None:
    node_names = _assert_cast_boundary_preserved(model, feeds, tmp_path, name)
    assert not any(name.startswith("MUSAExecutionProvider_") for name in node_names)


def test_cast_to_bfloat16_before_linear_pattern_preserves_cast_boundary(tmp_path):
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    w = float32_to_bfloat16_bits(
        np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)
    )
    bias = float32_to_bfloat16_bits(np.zeros((3,), dtype=np.float32))

    nodes = [
        helper.make_node("Cast", ["X"], ["X_bf16"], to=TensorProto.BFLOAT16),
        helper.make_node("MatMul", ["X_bf16", "W"], ["M"]),
        helper.make_node("Add", ["M", "Bias"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "cast_bf16_linear_boundary_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 3])],
        initializer=[
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(bias, "Bias"),
        ],
    )
    graph.initializer[0].data_type = TensorProto.BFLOAT16
    graph.initializer[1].data_type = TensorProto.BFLOAT16
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    node_names = _assert_cast_boundary_preserved(
        model.SerializeToString(),
        {"X": x},
        tmp_path,
        "cast_bf16_linear_boundary",
        outputs=[("Y", TensorProto.BFLOAT16, (2, 3))],
    )
    assert any("MatMulAddFusion" in name for name in node_names), node_names


def test_cast_to_float16_before_concat_matmul_preserves_cast_boundary(tmp_path):
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    x1 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    w = rng.standard_normal((2, 3, 9, 6)).astype(np.float16)

    nodes = [
        helper.make_node("Cast", ["X0"], ["X0_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["X1"], ["X1_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Concat", ["X0_f16", "X1_f16"], ["C"], axis=-1),
        helper.make_node("MatMul", ["C", "W"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "cast_f16_concat_matmul_boundary_graph",
        [
            helper.make_tensor_value_info("X0", TensorProto.FLOAT, list(x0.shape)),
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, list(x1.shape)),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3, 4, 6])],
        initializer=[numpy_helper.from_array(w, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    node_names = _assert_cast_boundary_preserved(
        model.SerializeToString(),
        {"X0": x0, "X1": x1},
        tmp_path,
        "cast_f16_concat_matmul_boundary",
    )
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
