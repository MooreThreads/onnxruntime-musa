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

    assert len(cpu_outputs) == len(musa_outputs), (
        f"{op_type}: output count mismatch cpu={len(cpu_outputs)} musa={len(musa_outputs)}"
    )
    for i, (c, m) in enumerate(zip(cpu_outputs, musa_outputs)):
        assert c.shape == m.shape, (
            f"{op_type} output[{i}] shape mismatch cpu={c.shape} musa={m.shape}"
        )
        assert c.dtype == m.dtype, (
            f"{op_type} output[{i}] dtype mismatch cpu={c.dtype} musa={m.dtype}"
        )
        np.testing.assert_allclose(
            m, c, rtol=rtol, atol=atol, err_msg=f"{op_type} output[{i}] CPU vs MUSA mismatch"
        )
    return musa_outputs


__all__ = [
    "TensorProto",
    "build_model",
    "musa_available",
    "musa_devices",
    "run",
    "run_and_compare",
]
