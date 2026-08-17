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

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_graph_model,
    float32_to_bfloat16_bits,
    musa_devices,
    run_with_iobinding,
)


def _profile_node_names(
    model: bytes,
    feeds: dict[str, np.ndarray],
    tmp_path,
    name: str,
    *,
    feed_types: dict[str, int] | None = None,
    outputs: list[tuple[str, int, tuple[int, ...]]] | None = None,
) -> list[str]:
    devices = musa_devices()
    if not devices:
        raise RuntimeError("FusedGemm fusion test requires a MUSA device")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / name)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model, sess_options=options)
    if outputs is None:
        session.run(None, feeds)
    else:
        io_binding = session.io_binding()
        feed_types = feed_types or {}
        for input_name, value in feeds.items():
            elem_type = feed_types.get(
                input_name, helper.np_dtype_to_tensor_dtype(value.dtype)
            )
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


def _assert_fused(
    model: bytes,
    feeds: dict[str, np.ndarray],
    tmp_path,
    name: str,
    *,
    feed_types: dict[str, int] | None = None,
    outputs: list[tuple[str, int, tuple[int, ...]]] | None = None,
) -> None:
    node_names = _profile_node_names(
        model, feeds, tmp_path, name, feed_types=feed_types, outputs=outputs
    )
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)


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


def test_matmul_add_tanh_fusion_float16(tmp_path):
    rng = np.random.default_rng(6)
    a = rng.standard_normal((2, 4, 16)).astype(np.float16)
    b = rng.standard_normal((16, 12)).astype(np.float16)
    bias = rng.standard_normal((12,)).astype(np.float16)

    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["M"]),
        helper.make_node("Add", ["M", "Bias"], ["MB"]),
        helper.make_node("Tanh", ["MB"], ["Y"]),
    ]
    feeds = {"A": a, "B": b, "Bias": bias}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT16)],
        name="matmul_add_tanh_float16_fusion_graph",
    )

    _run_fused_gemm_and_compare(model, feeds)
    _assert_fused(model, feeds, tmp_path, "matmul_add_tanh_float16")


def test_matmul_add_relu_fusion_bfloat16(tmp_path):
    rng = np.random.default_rng(7)
    a_f32 = rng.standard_normal((2, 4, 16)).astype(np.float32)
    b_f32 = rng.standard_normal((16, 12)).astype(np.float32)
    bias_f32 = rng.standard_normal((12,)).astype(np.float32)
    feeds = {
        "A": float32_to_bfloat16_bits(a_f32),
        "B": float32_to_bfloat16_bits(b_f32),
    }
    bias_bits = float32_to_bfloat16_bits(bias_f32)
    bias = numpy_helper.from_array(bias_bits, "Bias")
    bias.data_type = TensorProto.BFLOAT16

    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["M"]),
        helper.make_node("Add", ["M", "Bias"], ["MB"]),
        helper.make_node("Relu", ["MB"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "matmul_add_relu_bfloat16_fusion_graph",
        [
            helper.make_tensor_value_info("A", TensorProto.BFLOAT16, list(a_f32.shape)),
            helper.make_tensor_value_info("B", TensorProto.BFLOAT16, list(b_f32.shape)),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 4, 12])],
        initializer=[bias],
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model_proto.ir_version = min(model_proto.ir_version, 10)
    model = model_proto.SerializeToString()

    outputs = run_with_iobinding(
        model,
        feeds,
        {"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, (2, 4, 12))],
        use_musa=True,
    )
    expected = np.maximum(
        bfloat16_bits_to_float32(feeds["A"]) @ bfloat16_bits_to_float32(feeds["B"])
        + bfloat16_bits_to_float32(bias_bits),
        0.0,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(outputs[0]), expected, rtol=6e-2, atol=6e-2
    )
    _assert_fused(
        model,
        feeds,
        tmp_path,
        "matmul_add_relu_bfloat16",
        feed_types={"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16, (2, 4, 12))],
    )


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


def test_symbolic_residual_add_is_not_fused_as_bias():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((1, 4, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    residual = rng.standard_normal((1, 4, 12)).astype(np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["A", "B"], ["M"]),
            helper.make_node("Add", ["M", "Residual"], ["Y"]),
        ],
        "symbolic_residual_add_graph",
        [
            helper.make_tensor_value_info(
                "A", TensorProto.FLOAT, ["batch", "sequence", 16]
            ),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [16, 12]),
            helper.make_tensor_value_info(
                "Residual", TensorProto.FLOAT, ["batch", "sequence", 12]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, ["batch", "sequence", 12]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    _run_fused_gemm_and_compare(
        model.SerializeToString(), {"A": a, "B": b, "Residual": residual}
    )
