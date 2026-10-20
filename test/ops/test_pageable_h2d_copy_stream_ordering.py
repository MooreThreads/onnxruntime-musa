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
"""Small pageable host->device copies must be ordered on the caller's stream.

The plugin stages pageable host->device transfers of at least
`ORT_MUSA_PAGEABLE_H2D_BOUNCE_THRESHOLD_BYTES` (default 1024 B) through the
pinned host pool and issues them with `musaMemcpyAsync()` on the caller's
stream. Transfers below that threshold used to bypass the staging path and call
the blocking `musaMemcpy()` on the MUSA null stream instead, which is ordered
against neither the device work already queued on the EP's (non-blocking)
compute stream nor the consumers enqueued after it.

That missing order is the writer behind the `sherpa_whisper_base` /
`zipformer_encoder_en` failures: ORT's memory pattern hands the destination of a
small `MemcpyFromHost` (a 4/8/24 byte shape or scale tensor) the block of a
tensor that is already dead on the host side while its producer kernel is still
running on the device, so the copy read back the previous owner's bytes. See
`test_boundary_copy_ordering.py` for that end-to-end shape chain.

This test does not depend on the memory pattern placing the copy destination in
the path of a pending kernel. It drives the transfer directly - with
`onnxruntime.copy_tensors()`, the public entry point of `OrtApi::CopyTensors`
that the graph executor's MemcpyFromHost path uses as well - and queues the
pending writer on the same stream itself:

  1. `_PENDING_ROUNDS` device-side writers that keep the stream busy and, as the
     last one of every round, fill the destination bytes with a sentinel,
  2. the pageable host->device copy of those bytes, on that same stream,
  3. a stream synchronize and a read back of the destination.

The copy is the last writer enqueued on the stream, so the destination must hold
the copied bytes. If the copy ignores the stream, it runs as soon as the host
issues it - while the sentinel rounds are still queued - and the sentinel wins.
The sub-threshold sizes must therefore fail on a build that copies on the null
stream and pass on a build that stages the copy on the caller's stream. The
4096 B case is the control: it is above the staging threshold, so it was already
ordered before the fix.
"""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np
import onnxruntime as ort
import pytest

from op_test_utils import musa_devices

_MUSA_VENDOR_ID = 0x4D54

#: Production threshold of the staged (pinned pool + musaMemcpyAsync) H2D path.
_STAGED_THRESHOLD_BYTES = 1024

#: Device work queued on the stream ahead of the copy under test. Every round
#: keeps the stream busy; the copy must not be able to slip behind the last
#: round, so the rounds together have to outlast the host-side cost of one copy.
_PENDING_ROUNDS = 8
_PENDING_BYTES = 256 << 20
_SENTINEL_BYTE = 0xAB


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


class _MusaRuntime:
    """The MUSA runtime calls this test needs to queue work on an EP stream."""

    def __init__(self) -> None:
        self._lib = _load_musa_runtime()
        lib = self._lib
        lib.musaSetDevice.argtypes = [ctypes.c_int]
        lib.musaSetDevice.restype = ctypes.c_int
        lib.musaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.musaMalloc.restype = ctypes.c_int
        lib.musaFree.argtypes = [ctypes.c_void_p]
        lib.musaFree.restype = ctypes.c_int
        lib.musaMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.musaMemsetAsync.restype = ctypes.c_int
        lib.musaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.musaStreamSynchronize.restype = ctypes.c_int

        assert lib.musaSetDevice(0) == 0

    def malloc(self, nbytes: int) -> int:
        pointer = ctypes.c_void_p()
        assert self._lib.musaMalloc(ctypes.byref(pointer), nbytes) == 0
        assert pointer.value not in (None, 0)
        return pointer.value

    def free(self, pointer: int) -> None:
        if pointer:
            assert self._lib.musaFree(ctypes.c_void_p(pointer)) == 0

    def memset_async(self, pointer: int, value: int, nbytes: int, stream: int) -> None:
        assert (
            self._lib.musaMemsetAsync(
                ctypes.c_void_p(pointer),
                ctypes.c_int(value),
                ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream),
            )
            == 0
        )

    def synchronize(self, stream: int) -> None:
        assert self._lib.musaStreamSynchronize(ctypes.c_void_p(stream)) == 0


def _assert_copy_is_ordered(dtype, shape) -> None:
    devices = musa_devices()
    if not devices:
        pytest.skip("No MUSA device available")

    payload = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    nbytes = payload.nbytes
    # A plain CPU OrtValue: the source is pageable host memory, which is what
    # takes the path under test.
    source = ort.OrtValue.ortvalue_from_numpy(payload)
    target = ort.OrtValue.ortvalue_from_shape_and_type(
        shape, dtype, "gpu", 0, _MUSA_VENDOR_ID
    )
    assert target.data_ptr() != 0

    runtime = _MusaRuntime()
    stream = devices[0].create_sync_stream()
    handle = stream.get_handle()
    assert handle not in (None, 0)

    scratch = runtime.malloc(_PENDING_BYTES)
    try:
        # Warm up the device, the copy path and the pinned staging pool so that
        # the host-side cost of the copy below is the steady-state one.
        runtime.memset_async(scratch, 0, 4096, handle)
        runtime.synchronize(handle)
        ort.copy_tensors([source], [target], stream)
        runtime.synchronize(handle)

        for _ in range(_PENDING_ROUNDS):
            runtime.memset_async(scratch, 0, _PENDING_BYTES, handle)
            # Last writer queued before the copy: a copy that is not ordered on
            # this stream is overwritten by one of these fills.
            runtime.memset_async(target.data_ptr(), _SENTINEL_BYTE, nbytes, handle)

        ort.copy_tensors([source], [target], stream)
        runtime.synchronize(handle)

        got = np.asarray(target.numpy()).reshape(shape)
        np.testing.assert_array_equal(
            got.view(np.uint8).reshape(-1),
            payload.view(np.uint8).reshape(-1),
            err_msg=(
                f"{nbytes} byte pageable host->device copy to {shape} {np.dtype(dtype)}: "
                f"destination bytes changed after the copy; the copy was not ordered on the "
                f"caller's stream (a later device-side write clobbered it)"
            ),
        )
    finally:
        runtime.free(scratch)


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (np.int64, (3,)),      # 24 B
        (np.int32, (24,)),     # 96 B
        (np.float32, (128,)),  # 512 B
    ],
    ids=["24B", "96B", "512B"],
)
def test_small_pageable_h2d_copy_is_ordered_on_the_ep_stream(dtype, shape):
    assert np.prod(shape) * np.dtype(dtype).itemsize < _STAGED_THRESHOLD_BYTES
    _assert_copy_is_ordered(dtype, shape)


def test_pageable_h2d_copy_above_the_staging_threshold_is_ordered():
    shape = (1024,)  # 4096 B > threshold: staged on the caller's stream already
    assert np.prod(shape) * np.dtype(np.float32).itemsize >= _STAGED_THRESHOLD_BYTES
    _assert_copy_is_ordered(np.float32, shape)
