# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""External MUSA compute stream provider-option coverage."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import warnings

import numpy as np
import onnxruntime as ort
import pytest

from op_test_utils import TensorProto, build_model, musa_devices, run


def _load_musa_runtime():
    candidates = [
        "/usr/local/musa/lib/libmusart.so",
        "/usr/local/musa/lib/libmusart.so.5",
        ctypes.util.find_library("musart"),
        ctypes.util.find_library("musa"),
        ctypes.util.find_library("musa_runtime"),
        "libmusart.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    pytest.skip("MUSA runtime library is not loadable via ctypes")


class _MusaStream:
    def __init__(self):
        self._runtime = _load_musa_runtime()
        self._runtime.musaSetDevice.argtypes = [ctypes.c_int]
        self._runtime.musaSetDevice.restype = ctypes.c_int
        self._runtime.musaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._runtime.musaStreamCreate.restype = ctypes.c_int
        self._runtime.musaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._runtime.musaStreamSynchronize.restype = ctypes.c_int
        self._runtime.musaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._runtime.musaStreamDestroy.restype = ctypes.c_int

        assert self._runtime.musaSetDevice(0) == 0
        self.ptr = ctypes.c_void_p()
        assert self._runtime.musaStreamCreate(ctypes.byref(self.ptr)) == 0
        assert self.ptr.value not in (None, 0)

    def synchronize(self) -> None:
        assert self._runtime.musaStreamSynchronize(self.ptr) == 0

    def destroy(self) -> None:
        if self.ptr.value:
            assert self._runtime.musaStreamDestroy(self.ptr) == 0
            self.ptr = ctypes.c_void_p()


def _make_musa_session_with_options(model: bytes, ep_options: dict[str, str]):
    devices = musa_devices()
    if not devices:
        pytest.skip("No MUSA device available")

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(devices, ep_options)
    return ort.InferenceSession(model, sess_options=so)


def test_user_compute_stream_runs_add_and_remains_user_owned():
    devices = musa_devices()
    if not devices:
        pytest.skip("No MUSA device available")

    a = np.random.default_rng(101).standard_normal((4, 8)).astype(np.float32)
    b = np.random.default_rng(102).standard_normal((4, 8)).astype(np.float32)
    model = build_model("Add", {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])
    (expected,) = run(model, {"A": a, "B": b}, use_musa=False)

    stream = _MusaStream()
    session = _make_musa_session_with_options(
        model, {"user_compute_stream": str(stream.ptr.value)}
    )

    (actual,) = session.run(None, {"A": a, "B": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    del session
    gc.collect()

    stream.synchronize()
    stream.destroy()


def test_has_user_compute_stream_requires_non_null_stream():
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32)
    model = build_model("Add", {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Specified 'providers'/'provider_options' when creating "
            "InferenceSession.*",
            category=UserWarning,
        )
        with pytest.raises(Exception):
            _make_musa_session_with_options(
                model, {"has_user_compute_stream": "1", "user_compute_stream": "0"}
            )
