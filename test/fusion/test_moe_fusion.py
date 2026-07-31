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
"""End-to-end coverage for the single- and multi-router MoE fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from op_test_utils import musa_devices, run_model_and_compare


def _build_model(
    expert_count, *, rows=5, router_first=True, second_activation="Relu"
):
    rng = np.random.default_rng(20260731 + expert_count)
    input_width, hidden_width, output_width = 7, 11, 6
    x = rng.standard_normal((rows, input_width)).astype(np.float32)
    router = rng.random((rows, expert_count, 1), dtype=np.float32)
    router /= np.sum(router, axis=1, keepdims=True)

    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["axes"],
            name="MoE/axes",
            value=numpy_helper.from_array(np.array([1], dtype=np.int64)),
        )
    ]
    initializers = []
    expert_outputs = []
    for expert in range(expert_count):
        w1 = (rng.standard_normal((hidden_width, input_width)) / 4).astype(
            np.float32
        )
        b1 = (rng.standard_normal(hidden_width) / 8).astype(np.float32)
        w2 = (rng.standard_normal((output_width, hidden_width)) / 4).astype(
            np.float32
        )
        b2 = (rng.standard_normal(output_width) / 8).astype(np.float32)
        initializers.extend(
            [
                numpy_helper.from_array(w1, f"W1_{expert}"),
                numpy_helper.from_array(b1, f"B1_{expert}"),
                numpy_helper.from_array(w2, f"W2_{expert}"),
                numpy_helper.from_array(b2, f"B2_{expert}"),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Gemm",
                    ["X", f"W1_{expert}", f"B1_{expert}"],
                    [f"G1_{expert}"],
                    name=f"expert_{expert}/Gemm_0",
                    transB=1,
                ),
                helper.make_node(
                    "Relu",
                    [f"G1_{expert}"],
                    [f"R1_{expert}"],
                    name=f"expert_{expert}/Relu_0",
                ),
                helper.make_node(
                    "Gemm",
                    [f"R1_{expert}", f"W2_{expert}", f"B2_{expert}"],
                    [f"G2_{expert}"],
                    name=f"expert_{expert}/Gemm_1",
                    transB=1,
                ),
                helper.make_node(
                    second_activation,
                    [f"G2_{expert}"],
                    [f"R2_{expert}"],
                    name=f"expert_{expert}/{second_activation}_1",
                ),
                helper.make_node(
                    "Unsqueeze",
                    [f"R2_{expert}", "axes"],
                    [f"U_{expert}"],
                    name=f"expert_{expert}/Unsqueeze",
                ),
            ]
        )
        expert_outputs.append(f"U_{expert}")

    nodes.append(
        helper.make_node(
            "Concat", expert_outputs, ["Experts"], name="MoE/Concat", axis=-2
        )
    )
    mul_inputs = ["Router", "Experts"] if router_first else ["Experts", "Router"]
    nodes.extend(
        [
            helper.make_node("Mul", mul_inputs, ["Weighted"], name="MoE/Mul"),
            helper.make_node(
                "ReduceSum",
                ["Weighted", "axes"],
                ["Y"],
                name="MoE/ReduceSum",
                keepdims=0,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "short_term_moe_fusion_graph",
        [
            helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, ["rows", input_width]
            ),
            helper.make_tensor_value_info(
                "Router", TensorProto.FLOAT, ["rows", expert_count, 1]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, ["rows", output_width]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), {"X": x, "Router": router}


def _build_main_net_model(*, rows=4, expert_count=8, router_count=12):
    """Build the main-net MatMul + Add layout with shared expert outputs."""
    rng = np.random.default_rng(20260731 + expert_count + router_count)
    input_width, hidden_width, output_width = 9, 13, 7
    x = rng.standard_normal((1, rows, input_width)).astype(np.float32)

    nodes = []
    initializers = [
        numpy_helper.from_array(
            np.array([-1], dtype=np.int64), "UnsqueezeAxes"
        ),
        numpy_helper.from_array(
            np.array([3], dtype=np.int64), "ReduceAxes"
        ),
    ]
    expert_outputs = []
    for expert in range(expert_count):
        w1 = (rng.standard_normal((input_width, hidden_width)) / 4).astype(
            np.float32
        )
        b1 = (rng.standard_normal(hidden_width) / 8).astype(np.float32)
        w2 = (rng.standard_normal((hidden_width, output_width)) / 4).astype(
            np.float32
        )
        b2 = (rng.standard_normal(output_width) / 8).astype(np.float32)
        initializers.extend(
            [
                numpy_helper.from_array(w1, f"MainW1_{expert}"),
                numpy_helper.from_array(b1, f"MainB1_{expert}"),
                numpy_helper.from_array(w2, f"MainW2_{expert}"),
                numpy_helper.from_array(b2, f"MainB2_{expert}"),
            ]
        )
        first_add_inputs = (
            [f"MainMM1_{expert}", f"MainB1_{expert}"]
            if expert % 2 == 0
            else [f"MainB1_{expert}", f"MainMM1_{expert}"]
        )
        second_add_inputs = (
            [f"MainB2_{expert}", f"MainMM2_{expert}"]
            if expert % 2 == 0
            else [f"MainMM2_{expert}", f"MainB2_{expert}"]
        )
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    ["X", f"MainW1_{expert}"],
                    [f"MainMM1_{expert}"],
                    name=f"domain_experts/{expert}/MatMul_0",
                ),
                helper.make_node(
                    "Add",
                    first_add_inputs,
                    [f"MainAdd1_{expert}"],
                    name=f"domain_experts/{expert}/Add_0",
                ),
                helper.make_node(
                    "Relu",
                    [f"MainAdd1_{expert}"],
                    [f"MainRelu1_{expert}"],
                    name=f"domain_experts/{expert}/Relu_0",
                ),
                helper.make_node(
                    "MatMul",
                    [f"MainRelu1_{expert}", f"MainW2_{expert}"],
                    [f"MainMM2_{expert}"],
                    name=f"domain_experts/{expert}/MatMul_1",
                ),
                helper.make_node(
                    "Add",
                    second_add_inputs,
                    [f"MainAdd2_{expert}"],
                    name=f"domain_experts/{expert}/Add_1",
                ),
                helper.make_node(
                    "Relu",
                    [f"MainAdd2_{expert}"],
                    [f"MainRelu2_{expert}"],
                    name=f"domain_experts/{expert}/Relu_1",
                ),
                helper.make_node(
                    "Unsqueeze",
                    [f"MainRelu2_{expert}", "UnsqueezeAxes"],
                    [f"MainExpert_{expert}"],
                    name=f"domain_experts/{expert}/Unsqueeze",
                ),
            ]
        )
        expert_outputs.append(f"MainExpert_{expert}")

    nodes.append(
        helper.make_node(
            "Concat",
            expert_outputs,
            ["SharedExperts"],
            name="main_net/Concat_2",
            axis=-1,
        )
    )
    feeds = {"X": x}
    graph_inputs = [
        helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, [1, "rows", input_width]
        )
    ]
    graph_outputs = []
    for router_index in range(router_count):
        name = f"Router_{router_index}"
        router = rng.random((1, rows, 1, expert_count), dtype=np.float32)
        router /= np.sum(router, axis=-1, keepdims=True)
        feeds[name] = router
        graph_inputs.append(
            helper.make_tensor_value_info(
                name, TensorProto.FLOAT, [1, "rows", 1, expert_count]
            )
        )
        mul_inputs = (
            ["SharedExperts", name]
            if router_index % 2 == 0
            else [name, "SharedExperts"]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Mul",
                    mul_inputs,
                    [f"Weighted_{router_index}"],
                    name=f"main_net/Mul_{router_index}",
                ),
                helper.make_node(
                    "ReduceSum",
                    [f"Weighted_{router_index}", "ReduceAxes"],
                    [f"Y_{router_index}"],
                    name=f"main_net/ReduceSum_{router_index}",
                    keepdims=0,
                ),
            ]
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                f"Y_{router_index}",
                TensorProto.FLOAT,
                [1, "rows", output_width],
            )
        )

    graph = helper.make_graph(
        nodes,
        "main_net_multi_router_moe_fusion_graph",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


def _profile_node_events(model, feeds, tmp_path):
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "moe_fusion")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=options)
    session.run(None, feeds)
    path = Path(session.end_profiling())
    try:
        events = json.loads(path.read_text())
    finally:
        path.unlink(missing_ok=True)
    return [event for event in events if event.get("cat") == "Node"]


@pytest.mark.parametrize("expert_count", [2, 3, 8])
@pytest.mark.parametrize("router_first", [False, True])
def test_moe_fusion_single_router(expert_count, router_first, tmp_path):
    model, feeds = _build_model(expert_count, router_first=router_first)
    run_model_and_compare(model, feeds, rtol=2e-3, atol=2e-3)
    events = _profile_node_events(model, feeds, tmp_path)
    node_names = [event.get("name", "") for event in events]
    fused = [
        name for name in node_names if name.startswith("MUSAExecutionProvider_")
    ]
    assert len(fused) == 1
    assert not any(
        name.startswith(
            ("Gemm_", "Relu_", "Unsqueeze_", "Concat_", "Mul_", "ReduceSum_")
        )
        for name in node_names
    )


def test_moe_fusion_empty_rows(tmp_path):
    model, feeds = _build_model(2, rows=0)
    run_model_and_compare(model, feeds, rtol=2e-3, atol=2e-3)
    events = _profile_node_events(model, feeds, tmp_path)
    assert sum(
        event.get("name", "").startswith("MUSAExecutionProvider_")
        for event in events
    ) == 1


def test_moe_fusion_main_net_twelve_routers(tmp_path):
    model, feeds = _build_main_net_model(expert_count=8, router_count=12)
    assert len(feeds) - 1 == 12
    run_model_and_compare(model, feeds, rtol=2e-3, atol=2e-3)
    events = _profile_node_events(model, feeds, tmp_path)
    node_names = [event.get("name", "") for event in events]
    fused = [
        name for name in node_names if name.startswith("MUSAExecutionProvider_")
    ]
    assert len(fused) == 1
    assert not any(
        name.startswith(
            (
                "MatMul_",
                "Add_",
                "Relu_",
                "Unsqueeze_",
                "Concat_",
                "Mul_",
                "ReduceSum_",
            )
        )
        for name in node_names
    )


def test_moe_fusion_rejects_non_relu_second_layer(tmp_path):
    model, feeds = _build_model(2, second_activation="Tanh")
    run_model_and_compare(model, feeds, rtol=2e-3, atol=2e-3)
    events = _profile_node_events(model, feeds, tmp_path)
    op_names = {event.get("args", {}).get("op_name") for event in events}
    assert "Concat" in op_names
    assert "ReduceSum" in op_names
