# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""End-to-end coverage for the dynamic Slice -> Concat StridedView fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, musa_devices, run_model_and_compare


def _scalar_int64(name, value):
    return helper.make_tensor(name, TensorProto.INT64, [1], [value])


def _profile_musa_ops(model: bytes, feeds: dict[str, np.ndarray], tmp_path, prefix: str):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    ops = set()
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith(
            "_kernel_time"
        ):
            continue
        args = event.get("args", {})
        if args.get("provider") == "MUSAExecutionProvider":
            ops.add(args.get("op_name"))
    return ops


def _build_strided_view_model(a, b, segments, output_type):
    """[N*S, B, D] -> [S, B, N*D] for any N > 1."""
    nodes = [helper.make_node("MatMul", ["A", "B"], ["X"]),
             helper.make_node("Shape", ["X"], ["shape"]),
             helper.make_node("Gather", ["shape", "axis_zero"], ["length"], axis=0),
             helper.make_node("Add", ["length", "round_bias"], ["rounded"]),
             helper.make_node("Div", ["rounded", "segments"], ["block"])]
    initializers = [_scalar_int64("axis_zero", 0),
                    _scalar_int64("round_bias", segments - 1),
                    _scalar_int64("segments", segments),
                    _scalar_int64("zero", 0),
                    _scalar_int64("slice_axis", 0)]
    bounds = ["zero"]
    for i in range(1, segments + 1):
        factor = f"factor_{i}"
        bound = f"bound_{i}"
        initializers.append(_scalar_int64(factor, i))
        nodes.append(helper.make_node("Mul", ["block", factor], [bound]))
        bounds.append(bound)
    slice_outputs = []
    for i in range(segments):
        output = f"slice_{i}"
        nodes.append(helper.make_node("Slice", ["X", bounds[i], bounds[i + 1], "slice_axis"], [output]))
        slice_outputs.append(output)
    nodes.append(helper.make_node("Concat", slice_outputs, ["C"], axis=2))
    nodes.append(helper.make_node("Relu", ["C"], ["Y"]))
    return build_graph_model(
        nodes,
        {"A": a, "B": b},
        [("Y", output_type)],
        initializers=initializers,
        name="strided_view_dynamic_sequence_reorder",
    )


@pytest.mark.parametrize("segments", [3, 8])
def test_strided_view_fuses_dynamic_sequence_reorder(segments):
    rng = np.random.default_rng(17)
    a = rng.standard_normal((segments * 5, 2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 7)).astype(np.float32)
    model = _build_strided_view_model(a, b, segments, TensorProto.FLOAT)
    run_model_and_compare(model, {"A": a, "B": b}, rtol=1e-5, atol=1e-5)


def test_strided_view_fusion_float16_preserves_payload_and_fuses(tmp_path):
    segments = 4
    rng = np.random.default_rng(31)
    a = rng.standard_normal((segments * 3, 2, 5)).astype(np.float16)
    b = rng.standard_normal((5, 6)).astype(np.float16)
    model = _build_strided_view_model(a, b, segments, TensorProto.FLOAT16)

    run_model_and_compare(model, {"A": a, "B": b}, rtol=2e-2, atol=2e-2)
    musa_ops = _profile_musa_ops(model, {"A": a, "B": b}, tmp_path, "strided_view_fp16")
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "Slice" not in musa_ops
    assert "Concat" not in musa_ops
