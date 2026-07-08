# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""Device OrtValue IOBinding coverage for the MUSA EP."""

import numpy as np
import onnxruntime as ort

from op_test_utils import TensorProto, _make_session, build_model, run


_MUSA_VENDOR_ID = 0x4D54


def _musa_ortvalue_from_numpy(value: np.ndarray) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_numpy(value, "gpu", 0, _MUSA_VENDOR_ID)


def _empty_musa_ortvalue(shape: tuple[int, ...], dtype) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_shape_and_type(
        shape, dtype, "gpu", 0, _MUSA_VENDOR_ID
    )


def test_session_run_accepts_device_ortvalue_inputs():
    a = np.random.default_rng(2).standard_normal((8, 16)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((8, 16)).astype(np.float32)
    model = build_model("Add", {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])

    (expected,) = run(model, {"A": a, "B": b}, use_musa=False)
    session = _make_session(model, use_musa=True)

    a_device = _musa_ortvalue_from_numpy(a)
    b_device = _musa_ortvalue_from_numpy(b)

    (actual,) = session.run(None, {"A": a_device, "B": b_device})

    assert a_device.data_ptr() != 0
    assert b_device.data_ptr() != 0
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_device_ortvalue_input_and_output_iobinding_matches_cpu():
    a = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((8, 16)).astype(np.float32)
    model = build_model("Add", {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])

    (expected,) = run(model, {"A": a, "B": b}, use_musa=False)
    session = _make_session(model, use_musa=True)

    a_device = _musa_ortvalue_from_numpy(a)
    b_device = _musa_ortvalue_from_numpy(b)
    y_device = _empty_musa_ortvalue(tuple(expected.shape), expected.dtype)

    binding = session.io_binding()
    binding.bind_ortvalue_input("A", a_device)
    binding.bind_ortvalue_input("B", b_device)
    binding.bind_ortvalue_output("Y", y_device)

    session.run_with_iobinding(binding)
    binding.synchronize_outputs()

    assert a_device.data_ptr() != 0
    assert b_device.data_ptr() != 0
    assert y_device.data_ptr() != 0
    assert y_device.device_name() in {"cuda", "gpu"}
    np.testing.assert_allclose(y_device.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_hybrid_cpu_and_device_input_iobinding_matches_cpu():
    a = np.random.default_rng(4).standard_normal((8, 16)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((8, 16)).astype(np.float32)
    model = build_model("Add", {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])

    (expected,) = run(model, {"A": a, "B": b}, use_musa=False)
    session = _make_session(model, use_musa=True)

    a_device = _musa_ortvalue_from_numpy(a)
    y_device = _empty_musa_ortvalue(tuple(expected.shape), expected.dtype)

    binding = session.io_binding()
    binding.bind_ortvalue_input("A", a_device)
    binding.bind_cpu_input("B", b)
    binding.bind_ortvalue_output("Y", y_device)

    session.run_with_iobinding(binding)
    binding.synchronize_outputs()

    assert a_device.data_ptr() != 0
    assert y_device.data_ptr() != 0
    assert y_device.device_name() in {"cuda", "gpu"}
    np.testing.assert_allclose(y_device.numpy(), expected, rtol=1e-5, atol=1e-6)
