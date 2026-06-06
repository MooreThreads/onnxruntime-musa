# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Shared helpers for the end-to-end op tests.

Each test builds a single-node ONNX model, runs it on the stock CPU EP and on
the MUSA plugin EP, and checks that

  1. neither run raises (segfault / unregistered dtype / missing kernel surface
     as a Python exception or a hard crash), and
  2. the two providers produce numerically close outputs.

The MUSA EP is selected with CPU fallback disabled, so an unsupported op or
dtype fails loudly instead of silently running on CPU.
"""

from __future__ import annotations

import threading
from typing import Iterable, Mapping, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

_EP_NAME = "MUSAExecutionProvider"

_register_lock = threading.Lock()
_registered = False


def _ensure_registered() -> None:
    """Register the MUSA plugin EP library with ORT exactly once per process."""
    global _registered
    with _register_lock:
        if _registered:
            return
        import onnxruntime_musa as musa_ep

        try:
            ort.register_execution_provider_library(
                musa_ep.get_ep_name(), musa_ep.get_library_path()
            )
        except Exception:
            # Registering the same library twice in one process raises; ignore.
            pass
        _registered = True


def musa_devices() -> list:
    """Return the list of MUSA OrtEpDevice objects (empty if none/unavailable)."""
    try:
        _ensure_registered()
        return [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME]
    except Exception:
        return []


def musa_available() -> bool:
    """True when at least one MUSA device is advertised by the plugin EP."""
    return len(musa_devices()) > 0


def build_model(
    op_type: str,
    inputs: Mapping[str, np.ndarray],
    outputs: Sequence[tuple[str, int]],
    attrs: Mapping[str, object] | None = None,
    domain: str = "",
    opset: int = 17,
) -> bytes:
    """Serialize a single-node ONNX model.

    inputs:  name -> numpy array (all treated as runtime graph inputs)
    outputs: list of (name, onnx TensorProto elem type); shapes are left dynamic
    attrs:   node attributes
    domain:  "" for ai.onnx, "com.microsoft" for contrib ops
    """
    input_vis = [
        helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(arr.dtype), list(arr.shape)
        )
        for name, arr in inputs.items()
    ]
    output_vis = [helper.make_tensor_value_info(name, etype, None) for name, etype in outputs]

    node = helper.make_node(
        op_type,
        list(inputs.keys()),
        [name for name, _ in outputs],
        domain=domain,
        **(attrs or {}),
    )

    graph = helper.make_graph([node], f"{op_type}_graph", input_vis, output_vis)

    opset_imports = [helper.make_opsetid("", opset)]
    if domain:
        opset_imports.append(helper.make_opsetid(domain, 1))

    model = helper.make_model(graph, opset_imports=opset_imports)
    # Keep IR version within what ORT 1.26 accepts.
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def build_model_with_input_types(
    op_type: str,
    inputs: Mapping[str, np.ndarray],
    input_types: Mapping[str, int],
    outputs: Sequence[tuple[str, int]],
    attrs: Mapping[str, object] | None = None,
    domain: str = "",
    opset: int = 17,
) -> bytes:
    """Serialize a single-node ONNX model with explicit input ONNX dtypes."""
    input_vis = [
        helper.make_tensor_value_info(
            name,
            input_types.get(name, helper.np_dtype_to_tensor_dtype(arr.dtype)),
            list(arr.shape),
        )
        for name, arr in inputs.items()
    ]
    output_vis = [helper.make_tensor_value_info(name, etype, None) for name, etype in outputs]

    node = helper.make_node(
        op_type,
        list(inputs.keys()),
        [name for name, _ in outputs],
        domain=domain,
        **(attrs or {}),
    )

    graph = helper.make_graph([node], f"{op_type}_graph", input_vis, output_vis)
    opset_imports = [helper.make_opsetid("", opset)]
    if domain:
        opset_imports.append(helper.make_opsetid(domain, 1))

    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def build_graph_model(
    nodes: Sequence[onnx.NodeProto],
    inputs: Mapping[str, np.ndarray],
    outputs: Sequence[tuple[str, int]],
    *,
    initializers: Sequence[onnx.TensorProto] | None = None,
    opset: int = 17,
    name: str = "graph",
) -> bytes:
    """Serialize a small multi-node graph model for fusion pattern tests."""
    input_vis = [
        helper.make_tensor_value_info(
            input_name,
            helper.np_dtype_to_tensor_dtype(input_value.dtype),
            list(input_value.shape),
        )
        for input_name, input_value in inputs.items()
    ]
    output_vis = [
        helper.make_tensor_value_info(output_name, elem_type, None)
        for output_name, elem_type in outputs
    ]

    graph = helper.make_graph(
        list(nodes),
        name,
        input_vis,
        output_vis,
        initializer=list(initializers or []),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    # Keep IR version within what ORT 1.26 accepts.
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _make_session(model_bytes: bytes, use_musa: bool) -> ort.InferenceSession:
    if use_musa:
        devices = musa_devices()
        # Without a MUSA device, add_provider_for_devices([], ...) would silently
        # build a CPU-only session, turning run_and_compare into a meaningless
        # CPU-vs-CPU check. Fail loudly instead.
        if not devices:
            raise RuntimeError(
                "No MUSA device available; refusing to build a CPU-only session "
                "for the MUSA run (would compare CPU against CPU)."
            )
        so = ort.SessionOptions()
        # Disable CPU EP fallback so an op/dtype the MUSA EP does not support
        # errors loudly (at session init for unplaced nodes, or at run time for
        # unsupported kernels) instead of silently running on the CPU EP.
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(devices, {})
        return ort.InferenceSession(model_bytes, sess_options=so)
    return ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])


def run(model_bytes: bytes, feeds: Mapping[str, np.ndarray], use_musa: bool) -> list[np.ndarray]:
    """Create a session on the chosen EP and run it, returning the outputs."""
    session = _make_session(model_bytes, use_musa)
    return session.run(None, dict(feeds))


_ONNX_STORAGE_DTYPES = {
    TensorProto.BFLOAT16: np.uint16,
    TensorProto.BOOL: np.bool_,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.INT8: np.int8,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT8: np.uint8,
    TensorProto.UINT16: np.uint16,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
}


def float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 and return the raw uint16 payload."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bfloat16_bits_to_float32(values: np.ndarray) -> np.ndarray:
    """Convert raw BF16 uint16 payloads to float32 values."""
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def run_with_iobinding(
    model_bytes: bytes,
    feeds: Mapping[str, np.ndarray],
    feed_types: Mapping[str, int],
    outputs: Sequence[tuple[str, int, Sequence[int]]],
    *,
    use_musa: bool,
) -> list[np.ndarray]:
    """Run a model using raw buffers for ONNX dtypes without numpy dtypes."""
    session = _make_session(model_bytes, use_musa)
    io_binding = session.io_binding()
    for name, arr in feeds.items():
        elem_type = feed_types.get(name, helper.np_dtype_to_tensor_dtype(arr.dtype))
        io_binding.bind_input(name, "cpu", 0, elem_type, arr.shape, arr.ctypes.data)

    output_buffers = []
    for name, elem_type, shape in outputs:
        dtype = _ONNX_STORAGE_DTYPES[elem_type]
        output = np.empty(tuple(shape), dtype=dtype)
        io_binding.bind_output(name, "cpu", 0, elem_type, output.shape, output.ctypes.data)
        output_buffers.append(output)

    session.run_with_iobinding(io_binding)
    return output_buffers


def run_and_compare(
    op_type: str,
    *,
    inputs: Mapping[str, np.ndarray],
    outputs: Sequence[tuple[str, int]],
    attrs: Mapping[str, object] | None = None,
    domain: str = "",
    opset: int = 17,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> list[np.ndarray]:
    """Build the model, run it on CPU and MUSA, and assert outputs are close.

    Returns the MUSA outputs. Any exception (or hard crash) during either run
    propagates and fails the test, satisfying the "does it error?" check.
    """
    model_bytes = build_model(op_type, inputs, outputs, attrs, domain, opset)

    cpu_outputs = run(model_bytes, inputs, use_musa=False)
    musa_outputs = run(model_bytes, inputs, use_musa=True)

    assert len(cpu_outputs) == len(
        musa_outputs
    ), f"{op_type}: output count mismatch cpu={len(cpu_outputs)} musa={len(musa_outputs)}"
    for i, (c, m) in enumerate(zip(cpu_outputs, musa_outputs)):
        assert (
            c.shape == m.shape
        ), f"{op_type} output[{i}] shape mismatch cpu={c.shape} musa={m.shape}"
        assert (
            c.dtype == m.dtype
        ), f"{op_type} output[{i}] dtype mismatch cpu={c.dtype} musa={m.dtype}"
        np.testing.assert_allclose(
            m, c, rtol=rtol, atol=atol, err_msg=f"{op_type} output[{i}] CPU vs MUSA mismatch"
        )
    return musa_outputs


def run_model_and_compare(
    model_bytes: bytes,
    feeds: Mapping[str, np.ndarray],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> list[np.ndarray]:
    """Run a full ONNX model on CPU and MUSA, then compare all outputs."""
    cpu_outputs = run(model_bytes, feeds, use_musa=False)
    musa_outputs = run(model_bytes, feeds, use_musa=True)

    assert len(cpu_outputs) == len(
        musa_outputs
    ), f"model output count mismatch cpu={len(cpu_outputs)} musa={len(musa_outputs)}"
    for i, (cpu_output, musa_output) in enumerate(zip(cpu_outputs, musa_outputs)):
        assert (
            cpu_output.shape == musa_output.shape
        ), f"output[{i}] shape mismatch cpu={cpu_output.shape} musa={musa_output.shape}"
        assert (
            cpu_output.dtype == musa_output.dtype
        ), f"output[{i}] dtype mismatch cpu={cpu_output.dtype} musa={musa_output.dtype}"
        np.testing.assert_allclose(
            musa_output,
            cpu_output,
            rtol=rtol,
            atol=atol,
            err_msg=f"output[{i}] CPU vs MUSA mismatch",
        )
    return musa_outputs


def run_model_and_compare_with_cpu_fallback(
    model_bytes: bytes,
    feeds: Mapping[str, np.ndarray],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> list[np.ndarray]:
    """Run a full ONNX model on CPU and MUSA with ORT CPU fallback enabled."""
    cpu_outputs = run(model_bytes, feeds, use_musa=False)

    devices = musa_devices()
    if not devices:
        raise RuntimeError(
            "No MUSA device available; refusing to build a CPU-only session "
            "for the MUSA run (would compare CPU against CPU)."
        )
    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model_bytes, sess_options=so)
    musa_outputs = session.run(None, dict(feeds))

    assert len(cpu_outputs) == len(
        musa_outputs
    ), f"model output count mismatch cpu={len(cpu_outputs)} musa={len(musa_outputs)}"
    for i, (cpu_output, musa_output) in enumerate(zip(cpu_outputs, musa_outputs)):
        assert (
            cpu_output.shape == musa_output.shape
        ), f"output[{i}] shape mismatch cpu={cpu_output.shape} musa={musa_output.shape}"
        assert (
            cpu_output.dtype == musa_output.dtype
        ), f"output[{i}] dtype mismatch cpu={cpu_output.dtype} musa={musa_output.dtype}"
        np.testing.assert_allclose(
            musa_output,
            cpu_output,
            rtol=rtol,
            atol=atol,
            err_msg=f"output[{i}] CPU vs MUSA mismatch",
        )
    return musa_outputs


__all__ = [
    "TensorProto",
    "bfloat16_bits_to_float32",
    "build_graph_model",
    "build_model",
    "build_model_with_input_types",
    "float32_to_bfloat16_bits",
    "musa_available",
    "musa_devices",
    "run",
    "run_and_compare",
    "run_with_iobinding",
    "run_model_and_compare",
    "run_model_and_compare_with_cpu_fallback",
]
