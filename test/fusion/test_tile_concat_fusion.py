# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the Tile(identity) -> Concat fusion pattern."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, musa_devices, run_model_and_compare


def _profile_musa_session(model: bytes, feeds: dict[str, np.ndarray], tmp_path, prefix: str):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    outputs = session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    return outputs, events


def _node_events(events):
    return [
        event
        for event in events
        if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")
    ]


def _ops_by_provider(events):
    ops = {}
    for event in _node_events(events):
        args = event.get("args", {})
        ops.setdefault(args.get("provider"), set()).add(args.get("op_name"))
    return ops


def _assert_tile_concat_fused(events):
    ops_by_provider = _ops_by_provider(events)
    musa_ops = ops_by_provider.get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert fused_ops
    assert "Tile" not in musa_ops
    assert "Concat" not in musa_ops


def test_tile_identity_concat_fusion(tmp_path):
    rng = np.random.default_rng(17)
    x0 = rng.standard_normal((3, 2)).astype(np.float32)
    x1 = rng.standard_normal((3, 2)).astype(np.float32)
    x2 = rng.standard_normal((3, 1)).astype(np.float32)
    repeats = np.array([1, 1], dtype=np.int64)
    feeds = {"X0": x0, "X1": x1, "X2": x2, "repeats": repeats}
    model = build_graph_model(
        [
            helper.make_node("Tile", ["X0", "repeats"], ["T0"]),
            helper.make_node("Tile", ["X1", "repeats"], ["T1"]),
            helper.make_node("Concat", ["T0", "X2", "T1"], ["C"], axis=1),
            helper.make_node("Relu", ["C"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        name="tile_identity_concat_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    _, events = _profile_musa_session(model, feeds, tmp_path, "tile_identity_concat_fusion")
    _assert_tile_concat_fused(events)


def test_tile_concat_fusion_non_identity_repeats_fallback():
    rng = np.random.default_rng(19)
    x0 = rng.standard_normal((2, 2)).astype(np.float32)
    x1 = rng.standard_normal((2, 1)).astype(np.float32)
    repeats = np.array([1, 2], dtype=np.int64)
    feeds = {"X0": x0, "X1": x1, "repeats": repeats}
    model = build_graph_model(
        [
            helper.make_node("Tile", ["X0", "repeats"], ["T0"]),
            helper.make_node("Concat", ["T0", "X1"], ["C"], axis=1),
            helper.make_node("Relu", ["C"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        name="tile_concat_fusion_non_identity_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
