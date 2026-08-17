# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Session-scoped dtype contract diagnostics coverage."""

import numpy as np
import onnxruntime as ort

from op_test_utils import TensorProto, build_model, musa_devices


def test_dtype_diagnostics_reports_assignment_and_contract(capfd):
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for dtype diagnostic test")

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.ones((3, 4), dtype=np.float32)
    model = build_model("Add", {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(
        devices,
        {"precision_policy": "report", "dtype_diagnostics": "1"},
    )
    session = ort.InferenceSession(model, sess_options=so)
    (actual,) = session.run(None, {"A": a, "B": b})
    np.testing.assert_array_equal(actual, a + b)

    diagnostics = capfd.readouterr().err
    assert "MUSA_DTYPE_REPORT_BEGIN policy=report" in diagnostics
    assert "op=Add" in diagnostics
    assert "inputs=float32,float32" in diagnostics
    assert "assigned=1" in diagnostics
    assert "MUSA_DTYPE_REPORT_END" in diagnostics
