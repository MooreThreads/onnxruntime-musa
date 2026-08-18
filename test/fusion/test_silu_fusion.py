# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""End-to-end tests for the SiLU Plugin EP fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_graph_model,
    float32_to_bfloat16_bits,
    musa_devices,
    run,
    run_with_iobinding,
)


def _make_silu_nodes(swapped=False):
    mul_inputs = ["X", "S"] if not swapped else ["S", "X"]
    return [
        helper.make_node("Sigmoid", ["X"], ["S"]),
        helper.make_node("Mul", mul_inputs, ["Y"]),
    ]


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
        raise RuntimeError("SiLU fusion test requires a MUSA device")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / name)
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model, sess_options=options)
    if outputs is None:
        session.run(None, feeds)
    else:
        io_binding = session.io_binding()
        for input_name, value in feeds.items():
            elem_type = (feed_types or {}).get(
                input_name, helper.np_dtype_to_tensor_dtype(value.dtype)
            )
            io_binding.bind_input(
                input_name, "cpu", 0, elem_type, value.shape, value.ctypes.data
            )
        for output_name, elem_type, shape in outputs:
            output = np.empty(shape, dtype=np.uint16)
            io_binding.bind_output(
                output_name, "cpu", 0, elem_type, output.shape, output.ctypes.data
            )
        session.run_with_iobinding(io_binding)
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


def _assert_silu_fused(model: bytes, feeds: dict[str, np.ndarray], tmp_path, name: str):
    node_names = _profile_node_names(model, feeds, tmp_path, name)
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(name.startswith(("Sigmoid_", "Mul_")) for name in node_names)


def _silu_reference(x: np.ndarray, output_dtype) -> np.ndarray:
    value = x.astype(np.float64 if output_dtype == np.float64 else np.float32)
    result = value / (1.0 + np.exp(-value))
    return result.astype(output_dtype)


@pytest.mark.parametrize(
    "np_dtype,tensor_type,rtol,atol",
    [
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
        (np.float32, TensorProto.FLOAT, 1e-5, 1e-5),
        (np.float64, TensorProto.DOUBLE, 1e-12, 1e-12),
    ],
)
@pytest.mark.parametrize("swapped", [False, True])
def test_silu_fusion_float_dtypes(np_dtype, tensor_type, rtol, atol, swapped, tmp_path):
    x = np.array([[-8.0, -1.25, -0.0, 0.0, 1.25, 8.0]], dtype=np_dtype)
    model = build_graph_model(
        _make_silu_nodes(swapped=swapped),
        {"X": x},
        [("Y", tensor_type)],
        name="silu_fusion_graph",
    )

    (actual,) = run(model, {"X": x}, use_musa=True)
    np.testing.assert_allclose(
        actual,
        _silu_reference(x, np_dtype),
        rtol=rtol,
        atol=atol,
    )
    _assert_silu_fused(model, {"X": x}, tmp_path, f"silu_{np_dtype}_{swapped}")


def test_silu_fusion_bfloat16(tmp_path):
    x_f32 = np.array([[-8.0, -1.25, -0.0, 0.0, 1.25, 8.0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    reference_input = bfloat16_bits_to_float32(x)
    expected = float32_to_bfloat16_bits(
        reference_input / (1.0 + np.exp(-reference_input))
    )

    nodes = _make_silu_nodes()
    graph = helper.make_graph(
        nodes,
        "silu_bfloat16_fusion_graph",
        [helper.make_tensor_value_info("X", TensorProto.BFLOAT16, list(x.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    model_bytes = model.SerializeToString()

    (actual,) = run_with_iobinding(
        model_bytes,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        bfloat16_bits_to_float32(expected),
        rtol=1e-2,
        atol=2e-3,
    )

    node_names = _profile_node_names(
        model_bytes,
        {"X": x},
        tmp_path,
        "silu_bfloat16",
        feed_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16, x.shape)],
    )
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(name.startswith(("Sigmoid_", "Mul_")) for name in node_names)
