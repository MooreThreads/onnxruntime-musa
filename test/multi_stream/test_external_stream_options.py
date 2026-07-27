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
"""External MUSA compute stream provider-option coverage."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import warnings

import numpy as np
import onnxruntime as ort
import onnxruntime_musa as musa_ep
import pytest

from op_test_utils import TensorProto, build_model, musa_devices, run

_MUSA_VENDOR_ID = 0x4D54
_MUSA_MEMCPY_HOST_TO_DEVICE = 1
_MUSA_MEMCPY_DEVICE_TO_HOST = 2


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
        self._runtime.musaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._runtime.musaMemcpyAsync.restype = ctypes.c_int

        assert self._runtime.musaSetDevice(0) == 0
        self.ptr = ctypes.c_void_p()
        assert self._runtime.musaStreamCreate(ctypes.byref(self.ptr)) == 0
        assert self.ptr.value not in (None, 0)

    def synchronize(self) -> None:
        assert self._runtime.musaStreamSynchronize(self.ptr) == 0

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        assert (
            self._runtime.musaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                kind,
                self.ptr,
            )
            == 0
        )

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


def _musa_ortvalue_from_shape(shape: tuple[int, ...], dtype) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_shape_and_type(
        shape, dtype, "gpu", 0, _MUSA_VENDOR_ID
    )


def test_make_provider_options_for_user_compute_stream():
    stream = ctypes.c_void_p(0x1234)

    options = musa_ep.make_provider_options(user_compute_stream=stream)

    assert options["device_id"] == "0"
    assert options["has_user_compute_stream"] == "1"
    assert options["user_compute_stream"] == str(stream.value)
    assert options["use_ep_level_unified_stream"] == "1"
    assert options["do_copy_in_default_stream"] == "1"


def test_make_provider_options_rejects_null_user_compute_stream():
    with pytest.raises(ValueError):
        musa_ep.make_provider_options(user_compute_stream=0)


def test_make_provider_options_rejects_non_bool_options():
    with pytest.raises(TypeError):
        musa_ep.make_provider_options(do_copy_in_default_stream=2)


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
        model, musa_ep.make_provider_options(user_compute_stream=stream.ptr)
    )

    (actual,) = session.run(None, {"A": a, "B": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    del session
    gc.collect()

    stream.synchronize()
    stream.destroy()


def test_user_compute_stream_preserves_async_input_run_output_order():
    devices = musa_devices()
    if not devices:
        pytest.skip("No MUSA device available")

    a = np.random.default_rng(201).standard_normal((16, 16)).astype(np.float32)
    b = np.random.default_rng(202).standard_normal((16, 16)).astype(np.float32)
    expected = a + b
    model = build_model("Add", {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])

    stream = _MusaStream()
    session = _make_musa_session_with_options(
        model, musa_ep.make_provider_options(user_compute_stream=stream.ptr)
    )

    a_device = _musa_ortvalue_from_shape(tuple(a.shape), a.dtype)
    b_device = _musa_ortvalue_from_shape(tuple(b.shape), b.dtype)
    y_device = _musa_ortvalue_from_shape(tuple(expected.shape), expected.dtype)

    stream.memcpy_async(
        a_device.data_ptr(), a.ctypes.data, a.nbytes, _MUSA_MEMCPY_HOST_TO_DEVICE
    )
    stream.memcpy_async(
        b_device.data_ptr(), b.ctypes.data, b.nbytes, _MUSA_MEMCPY_HOST_TO_DEVICE
    )

    binding = session.io_binding()
    binding.bind_ortvalue_input("A", a_device)
    binding.bind_ortvalue_input("B", b_device)
    binding.bind_ortvalue_output("Y", y_device)

    session.run_with_iobinding(binding)

    actual = np.empty_like(expected)
    stream.memcpy_async(
        actual.ctypes.data,
        y_device.data_ptr(),
        actual.nbytes,
        _MUSA_MEMCPY_DEVICE_TO_HOST,
    )
    stream.synchronize()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    del session
    gc.collect()
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


def test_user_compute_stream_rejects_invalid_pointer_string():
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
                model, {"user_compute_stream": "not-a-pointer"}
            )


def test_boolean_provider_options_reject_non_boolean_values():
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
                model, {"do_copy_in_default_stream": "2"}
            )
