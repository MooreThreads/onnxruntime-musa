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
"""End-to-end tests for the four-branch parallel einsum activation fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _profile_musa_node_names(model_bytes, feeds):
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for profiling")

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = "parallel_einsum_activation_fusion"
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model_bytes, sess_options=session_options)
    session.run(None, dict(feeds))

    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


def _value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _build_model():
    rng = np.random.default_rng(20260707)
    batch = 3
    input_dim = 7
    hidden_dim = 5
    heads = 4
    feeds = {
        "X": rng.standard_normal((batch, input_dim, 1)).astype(np.float32),
        "G": rng.standard_normal((batch, input_dim, 1)).astype(np.float32),
        "B": rng.standard_normal((input_dim, 1)).astype(np.float32),
    }
    for head in range(heads):
        feeds[f"W1_{head}"] = rng.standard_normal((hidden_dim, input_dim)).astype(
            np.float32
        )
        feeds[f"W2_{head}"] = rng.standard_normal((hidden_dim, hidden_dim)).astype(
            np.float32
        )
        feeds[f"W3_{head}"] = rng.standard_normal((input_dim, hidden_dim)).astype(
            np.float32
        )

    nodes = []
    concat_inputs = []
    value_info = []
    for head in range(heads):
        e1 = f"E1_{head}"
        t1 = f"T1_{head}"
        e2 = f"E2_{head}"
        t2 = f"T2_{head}"
        e3 = f"E3_{head}"
        a = f"A_{head}"
        m = f"M_{head}"
        nodes.extend(
            [
                helper.make_node(
                    "Einsum",
                    [f"W1_{head}", "X"],
                    [e1],
                    equation="ij,bjk->bik",
                ),
                helper.make_node("Tanh", [e1], [t1]),
                helper.make_node(
                    "Einsum",
                    [f"W2_{head}", t1],
                    [e2],
                    equation="ij,bjk->bik",
                ),
                helper.make_node("Tanh", [e2], [t2]),
                helper.make_node(
                    "Einsum",
                    [f"W3_{head}", t2],
                    [e3],
                    equation="ij,bjk->bik",
                ),
                helper.make_node("Add", [e3, "B"], [a]),
                helper.make_node("Mul", ["G", a], [m]),
            ]
        )
        value_info.extend(
            [
                _value_info(e1, [batch, hidden_dim, 1]),
                _value_info(t1, [batch, hidden_dim, 1]),
                _value_info(e2, [batch, hidden_dim, 1]),
                _value_info(t2, [batch, hidden_dim, 1]),
                _value_info(e3, [batch, input_dim, 1]),
                _value_info(a, [batch, input_dim, 1]),
                _value_info(m, [batch, input_dim, 1]),
            ]
        )
        concat_inputs.append(m)

    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=2))
    nodes.append(helper.make_node("Identity", ["Y"], ["Z"]))
    value_info.append(_value_info("Y", [batch, input_dim, heads]))

    inputs = [
        _value_info("X", [batch, input_dim, 1]),
        _value_info("G", [batch, input_dim, 1]),
        _value_info("B", [input_dim, 1]),
    ]
    for head in range(heads):
        inputs.extend(
            [
                _value_info(f"W1_{head}", [hidden_dim, input_dim]),
                _value_info(f"W2_{head}", [hidden_dim, hidden_dim]),
                _value_info(f"W3_{head}", [input_dim, hidden_dim]),
            ]
        )

    graph = helper.make_graph(
        nodes,
        "parallel_einsum_activation_fusion_graph",
        inputs,
        [_value_info("Z", [batch, input_dim, heads])],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


def test_parallel_einsum_activation_fusion():
    model, feeds = _build_model()
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_musa_node_names(model, feeds)
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(
        name.startswith(("Einsum_", "Tanh_", "Add_", "Mul_", "Concat_"))
        for name in node_names
    )
