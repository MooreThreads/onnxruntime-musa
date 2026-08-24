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
"""End-to-end tests for shared-input parallel linear fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto as OnnxTensorProto, helper, numpy_helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    float32_to_bfloat16_bits,
    musa_devices,
    run_model_and_compare,
    run_with_iobinding,
)


def _build_model(
    branch_count, with_relu, bias_mask=None, output_widths=None, x_shape=None
):
    rng = np.random.default_rng(1700 + branch_count + int(with_relu))
    if x_shape is None:
        x_shape = (3, 4, 8)
    x = rng.standard_normal(x_shape).astype(np.float32)
    if bias_mask is None:
        bias_mask = [True] * branch_count
    if output_widths is None:
        output_widths = [5] * branch_count
    assert len(bias_mask) == branch_count
    assert len(output_widths) == branch_count
    nodes = []
    initializers = []
    outputs = []
    for i in range(branch_count):
        width = output_widths[i]
        weight = rng.standard_normal((8, width)).astype(np.float32)
        initializers.append(numpy_helper.from_array(weight, f"W{i}"))
        nodes.append(helper.make_node("MatMul", ["X", f"W{i}"], [f"M{i}"]))
        output = f"M{i}"
        if bias_mask[i]:
            bias = rng.standard_normal((width,)).astype(np.float32)
            initializers.append(numpy_helper.from_array(bias, f"B{i}"))
            output = f"A{i}"
            nodes.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [output]))
        if with_relu:
            relu_output = f"Y{i}"
            nodes.append(helper.make_node("Relu", [output], [relu_output]))
            output = relu_output
        outputs.append(
            helper.make_tensor_value_info(
                output, TensorProto.FLOAT, ["batch", 4, width]
            )
        )

    graph = helper.make_graph(
        nodes,
        "parallel_linear_fusion_graph",
        [
            helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, ["batch", 4, 8]
            )
        ],
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), {"X": x}


def _build_gated_mlp_model(with_bias):
    rng = np.random.default_rng(2718 + int(with_bias))
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    gate_weight = rng.standard_normal((8, 5)).astype(np.float32)
    up_weight = rng.standard_normal((8, 5)).astype(np.float32)
    initializers = [
        numpy_helper.from_array(gate_weight, "gate_weight"),
        numpy_helper.from_array(up_weight, "up_weight"),
    ]
    nodes = [
        helper.make_node("MatMul", ["X", "gate_weight"], ["gate_matmul"]),
        helper.make_node("MatMul", ["X", "up_weight"], ["up_matmul"]),
    ]
    gate = "gate_matmul"
    up = "up_matmul"
    if with_bias:
        gate_bias = rng.standard_normal((5,)).astype(np.float32)
        up_bias = rng.standard_normal((5,)).astype(np.float32)
        initializers.extend(
            [
                numpy_helper.from_array(gate_bias, "gate_bias"),
                numpy_helper.from_array(up_bias, "up_bias"),
            ]
        )
        gate = "gate"
        up = "up"
        nodes.extend(
            [
                helper.make_node("Add", ["gate_matmul", "gate_bias"], [gate]),
                helper.make_node("Add", ["up_matmul", "up_bias"], [up]),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Sigmoid", [gate], ["gate_sigmoid"]),
            helper.make_node("Mul", [gate, "gate_sigmoid"], ["gate_silu"]),
            helper.make_node("Mul", ["gate_silu", up], ["Y"]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "parallel_linear_gated_mlp_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 4, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 4, 5])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), {"X": x}


def _bf16_initializer(name, values):
    return helper.make_tensor(
        name,
        OnnxTensorProto.BFLOAT16,
        values.shape,
        values.astype(np.uint16).tobytes(),
        raw=True,
    )


def _build_bfloat16_qkv_model(branch_count=3):
    rng = np.random.default_rng(4096 + branch_count)
    x_f32 = rng.standard_normal((1, 5, 8)).astype(np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    weights_f32 = [
        rng.standard_normal((8, 8)).astype(np.float32)
        for _ in range(branch_count)
    ]
    weights = [float32_to_bfloat16_bits(weight) for weight in weights_f32]
    nodes = [
        helper.make_node("MatMul", ["X", f"W{i}"], [f"Y{i}"])
        for i in range(branch_count)
    ]
    graph = helper.make_graph(
        nodes,
        "parallel_linear_bfloat16_qkv_graph",
        [helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [1, 5, 8])],
        [
            helper.make_tensor_value_info(
                f"Y{i}", TensorProto.BFLOAT16, [1, 5, 8]
            )
            for i in range(branch_count)
        ],
        initializer=[
            _bf16_initializer(f"W{i}", weight)
            for i, weight in enumerate(weights)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    expected = [
        bfloat16_bits_to_float32(x) @ bfloat16_bits_to_float32(weight)
        for weight in weights
    ]
    return model.SerializeToString(), {"X": x}, expected


def _profile_node_names(model, feeds):
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for profiling")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = "parallel_linear_fusion"
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model, sess_options=options)
    session.run(None, feeds)
    path = Path(session.end_profiling())
    try:
        events = json.loads(path.read_text())
    finally:
        path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


@pytest.mark.parametrize("with_relu", [False, True])
def test_parallel_linear_fusion(with_relu):
    model, feeds = _build_model(branch_count=4, with_relu=with_relu)
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_node_names(model, feeds)
    fused = [
        name for name in node_names if name.startswith("MUSAExecutionProvider_")
    ]
    assert len(fused) == 1
    assert not any(
        name.startswith(("MatMul_", "Add_", "Relu_")) for name in node_names
    )


@pytest.mark.parametrize("with_relu", [False, True])
def test_parallel_linear_fusion_without_bias(with_relu):
    model, feeds = _build_model(
        branch_count=4, with_relu=with_relu, bias_mask=[False] * 4
    )
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_node_names(model, feeds)
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == 1
    assert not any(name.startswith(("MatMul_", "Relu_")) for name in node_names)


def test_parallel_linear_fusion_mixed_bias():
    model, feeds = _build_model(
        branch_count=4,
        with_relu=True,
        bias_mask=[True, False, True, False],
    )
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_node_names(model, feeds)
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == 1
    assert not any(
        name.startswith(("MatMul_", "Add_", "Relu_")) for name in node_names
    )


@pytest.mark.parametrize("branch_count", [2, 3])
def test_parallel_linear_fusion_direct_pointer_post(branch_count):
    model, feeds = _build_model(
        branch_count=branch_count,
        with_relu=True,
        bias_mask=[i % 2 == 0 for i in range(branch_count)],
    )
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_node_names(model, feeds)
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == 1
    assert not any(
        name.startswith(("MatMul_", "Add_", "Relu_")) for name in node_names
    )


@pytest.mark.parametrize("with_bias", [False, True])
def test_parallel_linear_gated_mlp_fusion(with_bias):
    model, feeds = _build_gated_mlp_model(with_bias)
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_node_names(model, feeds)
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == 1
    assert not any(
        name.startswith(("MatMul_", "Add_", "Sigmoid_", "Mul_"))
        for name in node_names
    )


def test_parallel_linear_fusion_empty_input():
    model, feeds = _build_model(
        branch_count=4, with_relu=True, x_shape=(0, 4, 8)
    )
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_node_names(model, feeds)
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == 1


def test_parallel_linear_fusion_matches_nine_of_ten_branches():
    model, feeds = _build_model(
        branch_count=10,
        with_relu=True,
        bias_mask=[True] * 9 + [False],
        output_widths=[5] * 9 + [7],
    )
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_node_names(model, feeds)
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == 1
    remaining_matmuls = [
        name for name in node_names if name.startswith("MatMul_")
    ]
    assert len(remaining_matmuls) == 1


def test_parallel_linear_bfloat16_qkv_fusion(tmp_path):
    model, feeds, expected = _build_bfloat16_qkv_model()
    outputs = [(f"Y{i}", TensorProto.BFLOAT16, (1, 5, 8)) for i in range(3)]
    actual = run_with_iobinding(
        model,
        feeds,
        {"X": TensorProto.BFLOAT16},
        outputs,
        use_musa=True,
    )
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(
            bfloat16_bits_to_float32(got), want, rtol=3e-2, atol=3e-2
        )

    devices = musa_devices()
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "parallel_linear_bfloat16_qkv")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model, sess_options=options)
    binding = session.io_binding()
    x = feeds["X"]
    binding.bind_input(
        "X", "cpu", 0, TensorProto.BFLOAT16, x.shape, x.ctypes.data
    )
    output_buffers = []
    for name, elem_type, shape in outputs:
        output = np.empty(shape, dtype=np.uint16)
        binding.bind_output(
            name, "cpu", 0, elem_type, output.shape, output.ctypes.data
        )
        output_buffers.append(output)
    session.run_with_iobinding(binding)
    path = Path(session.end_profiling())
    events = json.loads(path.read_text())
    path.unlink(missing_ok=True)
    node_names = [
        event.get("name", "") for event in events if event.get("cat") == "Node"
    ]
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(name.startswith("MatMul_") for name in node_names)


@pytest.mark.parametrize("branch_count", [2, 3])
def test_parallel_linear_bfloat16_grouped_direct_output(branch_count):
    model, feeds, expected = _build_bfloat16_qkv_model(branch_count)
    outputs = [
        (f"Y{i}", TensorProto.BFLOAT16, (1, 5, 8))
        for i in range(branch_count)
    ]
    actual = run_with_iobinding(
        model,
        feeds,
        {"X": TensorProto.BFLOAT16},
        outputs,
        use_musa=True,
    )
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(
            bfloat16_bits_to_float32(got), want, rtol=3e-2, atol=3e-2
        )
