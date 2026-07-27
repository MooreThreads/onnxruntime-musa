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
"""End-to-end tests for Concat -> Reshape fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    musa_devices,
    run_model_and_compare,
    run_model_and_compare_with_cpu_fallback,
)


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


def test_concat_reshape_fusion_with_fanout(tmp_path):
    rng = np.random.default_rng(29)
    feeds = {
        "X0": rng.standard_normal((2, 3, 4)).astype(np.float32),
        "X1": rng.standard_normal((2, 3, 4)).astype(np.float32),
    }
    weight0 = rng.standard_normal((8, 5)).astype(np.float32)
    weight1 = rng.standard_normal((8, 7)).astype(np.float32)
    target_shape = np.array([6, 8], dtype=np.int64)

    model = build_graph_model(
        [
            helper.make_node("Concat", ["X0", "X1"], ["C"], axis=2),
            helper.make_node("Reshape", ["C", "target_shape"], ["R"]),
            helper.make_node("MatMul", ["R", "W0"], ["Y0"]),
            helper.make_node("MatMul", ["R", "W1"], ["Y1"]),
        ],
        inputs=feeds,
        outputs=[("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(target_shape, name="target_shape"),
            numpy_helper.from_array(weight0, name="W0"),
            numpy_helper.from_array(weight1, name="W1"),
        ],
        name="concat_reshape_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    _, events = _profile_musa_session(model, feeds, tmp_path, "concat_reshape_fusion")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert fused_ops
    assert "Concat" not in musa_ops
    assert "Reshape" not in musa_ops


def test_concat_unsqueeze_reshape_fusion(tmp_path):
    rng = np.random.default_rng(31)
    feeds = {
        "X0": rng.standard_normal((2, 3, 4)).astype(np.float32),
        "X1": rng.standard_normal((2, 3, 2)).astype(np.float32),
    }
    weight = rng.standard_normal((6, 5)).astype(np.float32)
    axes = np.array([1], dtype=np.int64)
    target_shape = np.array([6, 6], dtype=np.int64)

    model = build_graph_model(
        [
            helper.make_node("Concat", ["X0", "X1"], ["C"], axis=2),
            helper.make_node("Unsqueeze", ["C", "axes"], ["U"]),
            helper.make_node("Reshape", ["U", "target_shape"], ["R"]),
            helper.make_node("MatMul", ["R", "W"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(axes, name="axes"),
            numpy_helper.from_array(target_shape, name="target_shape"),
            numpy_helper.from_array(weight, name="W"),
        ],
        name="concat_unsqueeze_reshape_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    _, events = _profile_musa_session(
        model, feeds, tmp_path, "concat_unsqueeze_reshape_fusion"
    )
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert fused_ops
    assert "Concat" not in musa_ops
    assert "Unsqueeze" not in musa_ops
    assert "Reshape" not in musa_ops


def test_concat_reshape_fusion_accepts_cpu_produced_input(tmp_path):
    rng = np.random.default_rng(41)
    feeds = {
        "X0": rng.standard_normal((2, 2)).astype(np.float32),
        "X1": rng.standard_normal((2, 2)).astype(np.float32),
    }
    weight = rng.standard_normal((4, 3)).astype(np.float32)
    target_shape = np.array([2, 4], dtype=np.int64)

    model = build_graph_model(
        [
            helper.make_node("Sin", ["X1"], ["S"]),
            helper.make_node("Concat", ["X0", "S"], ["C"], axis=1),
            helper.make_node("Reshape", ["C", "target_shape"], ["R"]),
            helper.make_node("MatMul", ["R", "W"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(target_shape, name="target_shape"),
            numpy_helper.from_array(weight, name="W"),
        ],
        name="concat_reshape_cpu_produced_input_graph",
    )

    run_model_and_compare_with_cpu_fallback(model, feeds, rtol=1e-5, atol=1e-5)
    _, events = _profile_musa_session(
        model, feeds, tmp_path, "concat_reshape_cpu_produced_input"
    )
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert fused_ops
    assert "Concat" not in musa_ops
    assert "Reshape" not in musa_ops


def test_concat_reshape_skips_constant_data_input():
    rng = np.random.default_rng(37)
    feeds = {
        "X": rng.standard_normal((2, 2)).astype(np.float32),
    }
    const_input = rng.standard_normal((2, 2)).astype(np.float32)
    weight = rng.standard_normal((4, 3)).astype(np.float32)
    target_shape = np.array([2, 4], dtype=np.int64)

    model = build_graph_model(
        [
            helper.make_node(
                "Constant",
                [],
                ["C"],
                value=numpy_helper.from_array(const_input),
            ),
            helper.make_node("Concat", ["X", "C"], ["ConcatOut"], axis=0),
            helper.make_node("Reshape", ["ConcatOut", "target_shape"], ["R"]),
            helper.make_node("MatMul", ["R", "W"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(target_shape, name="target_shape"),
            numpy_helper.from_array(weight, name="W"),
        ],
        name="concat_reshape_constant_data_input_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
