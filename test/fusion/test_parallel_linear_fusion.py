# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for shared-input parallel linear fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


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
