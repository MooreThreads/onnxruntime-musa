# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""End-to-end tests for projection + SwiGLU fusion."""

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
    run,
    run_with_iobinding,
)


def _bf16_initializer(name, values):
    return helper.make_tensor(
        name,
        OnnxTensorProto.BFLOAT16,
        values.shape,
        values.astype(np.uint16).tobytes(),
        raw=True,
    )


def _make_block_nodes(prefix, x_name, gate_weight, up_weight, *, swapped=False):
    gate = f"{prefix}_gate"
    up = f"{prefix}_up"
    sigmoid = f"{prefix}_sigmoid"
    silu = f"{prefix}_silu"
    output = f"{prefix}_output"
    gate_mul_inputs = [gate, sigmoid] if not swapped else [sigmoid, gate]
    output_mul_inputs = [silu, up] if not swapped else [up, silu]
    return [
        helper.make_node(
            "MatMul", [x_name, gate_weight], [gate], name=f"{prefix}_gate_proj"
        ),
        helper.make_node(
            "MatMul", [x_name, up_weight], [up], name=f"{prefix}_up_proj"
        ),
        helper.make_node("Sigmoid", [gate], [sigmoid], name=f"{prefix}_sigmoid"),
        helper.make_node("Mul", gate_mul_inputs, [silu], name=f"{prefix}_silu"),
        helper.make_node(
            "Mul", output_mul_inputs, [output], name=f"{prefix}_swiglu"
        ),
    ], output


def _build_model(np_dtype, tensor_type, *, blocks=1, swapped=False, empty=False):
    rng = np.random.default_rng(8128 + blocks + int(swapped))
    rows = 0 if empty else 5
    # Keep the eight-block reference in a numerically well-conditioned range;
    # this test is checking graph selection and device results, not overflow
    # handling in NumPy's reference sigmoid.
    x_f32 = (0.2 * rng.standard_normal((1, rows, 8))).astype(np.float32)
    x = (
        float32_to_bfloat16_bits(x_f32)
        if tensor_type == TensorProto.BFLOAT16
        else x_f32.astype(np_dtype)
    )
    nodes = []
    initializers = []
    reference = (
        bfloat16_bits_to_float32(x)
        if tensor_type == TensorProto.BFLOAT16
        else x.astype(np.float32)
    )
    current = "X"
    for block in range(blocks):
        gate_f32 = (0.2 * rng.standard_normal((8, 8))).astype(np.float32)
        up_f32 = (0.2 * rng.standard_normal((8, 8))).astype(np.float32)
        if tensor_type == TensorProto.BFLOAT16:
            gate = float32_to_bfloat16_bits(gate_f32)
            up = float32_to_bfloat16_bits(up_f32)
            initializers.extend(
                [
                    _bf16_initializer(f"Wg{block}", gate),
                    _bf16_initializer(f"Wu{block}", up),
                ]
            )
            gate_ref = reference @ bfloat16_bits_to_float32(gate)
            up_ref = reference @ bfloat16_bits_to_float32(up)
            gate_ref = bfloat16_bits_to_float32(
                float32_to_bfloat16_bits(gate_ref)
            )
            up_ref = bfloat16_bits_to_float32(float32_to_bfloat16_bits(up_ref))
        else:
            gate = gate_f32.astype(np_dtype)
            up = up_f32.astype(np_dtype)
            initializers.extend(
                [
                    numpy_helper.from_array(gate, f"Wg{block}"),
                    numpy_helper.from_array(up, f"Wu{block}"),
                ]
            )
            gate_ref = (reference @ gate.astype(np.float32)).astype(np_dtype)
            up_ref = (reference @ up.astype(np.float32)).astype(np_dtype)
            gate_ref = gate_ref.astype(np.float32)
            up_ref = up_ref.astype(np.float32)

        block_nodes, current = _make_block_nodes(
            f"block{block}", current, f"Wg{block}", f"Wu{block}", swapped=swapped
        )
        nodes.extend(block_nodes)
        reference = (gate_ref / (1.0 + np.exp(-gate_ref))) * up_ref
        if tensor_type == TensorProto.BFLOAT16:
            reference = bfloat16_bits_to_float32(
                float32_to_bfloat16_bits(reference)
            )
        else:
            reference = reference.astype(np_dtype).astype(np.float32)

    graph = helper.make_graph(
        nodes,
        "swiglu_fusion_graph",
        [helper.make_tensor_value_info("X", tensor_type, list(x.shape))],
        [helper.make_tensor_value_info(current, tensor_type, list(x.shape))],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), {"X": x}, current, reference


def _profile_node_names(model, feeds, tmp_path, name, *, tensor_type):
    devices = musa_devices()
    if not devices:
        raise RuntimeError("SwiGlu fusion test requires a MUSA device")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / name)
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model, sess_options=options)
    if tensor_type == TensorProto.BFLOAT16:
        binding = session.io_binding()
        x = feeds["X"]
        binding.bind_input(
            "X", "cpu", 0, TensorProto.BFLOAT16, x.shape, x.ctypes.data
        )
        output_name = session.get_outputs()[0].name
        output = np.empty(x.shape, dtype=np.uint16)
        binding.bind_output(
            output_name,
            "cpu",
            0,
            TensorProto.BFLOAT16,
            output.shape,
            output.ctypes.data,
        )
        session.run_with_iobinding(binding)
    else:
        session.run(None, feeds)
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


def _assert_fusion_count(node_names, count):
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == count
    assert not any(
        name.startswith(("MatMul_", "Sigmoid_", "Mul_")) for name in node_names
    )


@pytest.mark.parametrize(
    "np_dtype,tensor_type,rtol,atol",
    [
        (np.float16, TensorProto.FLOAT16, 3e-2, 3e-2),
        (np.float32, TensorProto.FLOAT, 2e-4, 2e-4),
    ],
)
@pytest.mark.parametrize("swapped", [False, True])
def test_swiglu_fusion_float_dtypes(
    np_dtype, tensor_type, rtol, atol, swapped, tmp_path
):
    model, feeds, _, expected = _build_model(
        np_dtype, tensor_type, swapped=swapped
    )
    (actual,) = run(model, feeds, use_musa=True)
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=rtol, atol=atol)
    node_names = _profile_node_names(
        model, feeds, tmp_path, f"swiglu_{np_dtype}_{swapped}", tensor_type=tensor_type
    )
    _assert_fusion_count(node_names, 1)


def test_swiglu_fusion_bfloat16(tmp_path):
    model, feeds, output_name, expected = _build_model(
        np.uint16, TensorProto.BFLOAT16
    )
    x = feeds["X"]
    (actual,) = run_with_iobinding(
        model,
        feeds,
        {"X": TensorProto.BFLOAT16},
        [(output_name, TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), expected, rtol=3e-2, atol=3e-2
    )
    node_names = _profile_node_names(
        model, feeds, tmp_path, "swiglu_bfloat16", tensor_type=TensorProto.BFLOAT16
    )
    _assert_fusion_count(node_names, 1)


def test_swiglu_fusion_hits_eight_unirank_blocks(tmp_path):
    model, feeds, _, expected = _build_model(
        np.float32, TensorProto.FLOAT, blocks=8
    )
    (actual,) = run(model, feeds, use_musa=True)
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)
    node_names = _profile_node_names(
        model, feeds, tmp_path, "swiglu_eight_blocks", tensor_type=TensorProto.FLOAT
    )
    _assert_fusion_count(node_names, 8)


def test_swiglu_fusion_hits_eight_bfloat16_blocks(tmp_path):
    model, feeds, output_name, expected = _build_model(
        np.uint16, TensorProto.BFLOAT16, blocks=8
    )
    x = feeds["X"]
    (actual,) = run_with_iobinding(
        model,
        feeds,
        {"X": TensorProto.BFLOAT16},
        [(output_name, TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), expected, rtol=3e-2, atol=3e-2
    )
    node_names = _profile_node_names(
        model, feeds, tmp_path, "swiglu_eight_bfloat16_blocks",
        tensor_type=TensorProto.BFLOAT16
    )
    _assert_fusion_count(node_names, 8)


def test_swiglu_fusion_empty_input(tmp_path):
    model, feeds, _, expected = _build_model(
        np.float32, TensorProto.FLOAT, empty=True
    )
    (actual,) = run(model, feeds, use_musa=True)
    assert actual.shape == expected.shape == (1, 0, 8)
    node_names = _profile_node_names(
        model, feeds, tmp_path, "swiglu_empty", tensor_type=TensorProto.FLOAT
    )
    _assert_fusion_count(node_names, 1)
