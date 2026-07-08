# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end test for the Split -> Reduce fusion pattern."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, musa_devices, run_model_and_compare


def _profile_musa_ops(model, feeds, tmp_path, prefix):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    session.run(None, dict(feeds))
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)

    ops = set()
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        args = event.get("args", {})
        if args.get("provider") == "MUSAExecutionProvider":
            ops.add(args.get("op_name"))
    return ops


def test_split_reduce_prod_mean_fusion():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.5, 1.5, size=(4, 5, 3)).astype(np.float32)
    split = helper.make_tensor("split", TensorProto.INT64, [2], [2, 3])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Split", ["X", "split"], ["S0", "S1"], axis=1),
        helper.make_node("ReduceProd", ["S0", "axes"], ["Y0"], keepdims=0),
        helper.make_node("ReduceMean", ["S1", "axes"], ["Y1"], keepdims=0),
    ]
    feeds = {"X": x}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        initializers=[split, axes],
        opset=18,
        name="split_reduce_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)


def test_split_reduce_three_way_fusion(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.uniform(0.5, 1.5, size=(4, 9, 3)).astype(np.float32)
    split = helper.make_tensor("split", TensorProto.INT64, [3], [2, 3, 4])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Split", ["X", "split"], ["S0", "S1", "S2"], axis=1),
        helper.make_node("ReduceProd", ["S0", "axes"], ["Y0"], keepdims=0),
        helper.make_node("ReduceMean", ["S1", "axes"], ["Y1"], keepdims=0),
        helper.make_node("ReduceProd", ["S2", "axes"], ["Y2"], keepdims=0),
    ]
    feeds = {"X": x}
    model = build_graph_model(
        nodes,
        feeds,
        [
            ("Y0", TensorProto.FLOAT),
            ("Y1", TensorProto.FLOAT),
            ("Y2", TensorProto.FLOAT),
        ],
        initializers=[split, axes],
        opset=18,
        name="split_reduce_three_way_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    musa_ops = _profile_musa_ops(model, feeds, tmp_path, "split_reduce_three_way")
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "Split" not in musa_ops
    assert "ReduceProd" not in musa_ops
    assert "ReduceMean" not in musa_ops
