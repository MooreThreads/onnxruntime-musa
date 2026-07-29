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
"""End-to-end tests for MHTA scaled dot-product attention fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_graph_model,
    float32_to_bfloat16_bits,
    musa_devices,
    run_model_and_compare,
    run_with_iobinding,
)


def _profile_musa_session(model: bytes, feeds: dict[str, np.ndarray], tmp_path, prefix: str):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
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


def _ops_by_provider(events):
    ops = {}
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        args = event.get("args", {})
        ops.setdefault(args.get("provider"), set()).add(args.get("op_name"))
    return ops


def _mhta_bhsd_nodes():
    return [
        helper.make_node("MatMul", ["Q", "K"], ["Score"]),
        helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
        helper.make_node("Add", ["Scaled", "zero_mask"], ["Masked"]),
        helper.make_node("Div", ["Masked", "temperature"], ["TempScaled"]),
        helper.make_node("Softmax", ["TempScaled"], ["Prob"], axis=-1),
        helper.make_node("MatMul", ["Prob", "V"], ["Y"]),
    ]


def _build_bfloat16_mhta_model(feeds, scale, temperature, zero_mask):
    def bf16_initializer(name, values):
        return helper.make_tensor(
            name,
            TensorProto.BFLOAT16,
            values.shape,
            values.astype(np.uint16).tobytes(),
            raw=True,
        )

    input_vis = [
        helper.make_tensor_value_info(name, TensorProto.BFLOAT16, value.shape)
        for name, value in feeds.items()
    ]
    graph = helper.make_graph(
        _mhta_bhsd_nodes(),
        "mhta_scaled_dot_product_attention_bfloat16_graph",
        input_vis,
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, None)],
        initializer=[
            bf16_initializer("scale", scale),
            bf16_initializer("temperature", temperature),
            bf16_initializer("zero_mask", zero_mask),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _reference_sdpa(q, k, v, scale, temperature):
    score = np.matmul(q, k) * scale / temperature
    score = score - np.max(score, axis=-1, keepdims=True)
    prob = np.exp(score)
    prob /= np.sum(prob, axis=-1, keepdims=True)
    return np.matmul(prob, v)


def test_mhta_scaled_dot_product_attention_fusion(tmp_path):
    rng = np.random.default_rng(41)
    batch, heads, seqlen, head_dim = 2, 3, 4, 5
    feeds = {
        "Q": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32),
        "K": rng.standard_normal((batch, heads, head_dim, seqlen)).astype(np.float32),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32),
    }
    scale = np.array(0.5, dtype=np.float32)
    temperature = np.array(2.0, dtype=np.float32)
    zero_mask = np.zeros((batch, heads, seqlen, seqlen), dtype=np.float32)

    model = build_graph_model(
        [
            helper.make_node("MatMul", ["Q", "K"], ["Score"]),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Add", ["Scaled", "zero_mask"], ["Masked"]),
            helper.make_node("Div", ["Masked", "temperature"], ["TempScaled"]),
            helper.make_node("Softmax", ["TempScaled"], ["Prob"], axis=-1),
            helper.make_node("MatMul", ["Prob", "V"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(temperature, name="temperature"),
            numpy_helper.from_array(zero_mask, name="zero_mask"),
        ],
        name="mhta_scaled_dot_product_attention_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    _, events = _profile_musa_session(model, feeds, tmp_path, "mhta_sdpa_fusion")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert fused_ops
    assert "Softmax" not in musa_ops
    assert "Div" not in musa_ops
    assert "Mul" not in musa_ops


def test_mhta_scaled_dot_product_attention_fp16_runflash(tmp_path):
    """FP16 must take the muDNN RunFlash branch and remain fused."""
    rng = np.random.default_rng(45)
    batch, heads, seqlen, head_dim = 1, 2, 4, 8
    feeds = {
        "Q": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float16),
        "K": rng.standard_normal((batch, heads, head_dim, seqlen)).astype(np.float16),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float16),
    }
    scale = np.array(0.25, dtype=np.float16)
    temperature = np.array(2.0, dtype=np.float16)
    zero_mask = np.zeros((batch, heads, seqlen, seqlen), dtype=np.float16)
    model = build_graph_model(
        _mhta_bhsd_nodes(),
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT16)],
        initializers=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(temperature, name="temperature"),
            numpy_helper.from_array(zero_mask, name="zero_mask"),
        ],
        name="mhta_scaled_dot_product_attention_fp16_graph",
    )

    run_model_and_compare(model, feeds, rtol=2e-2, atol=2e-2)
    _, events = _profile_musa_session(model, feeds, tmp_path, "mhta_sdpa_fp16")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "Softmax" not in musa_ops


def test_mhta_scaled_dot_product_attention_bfloat16_runflash():
    """BF16 raw buffers exercise the same RunFlash dtype dispatch."""
    rng = np.random.default_rng(46)
    batch, heads, seqlen, head_dim = 1, 2, 4, 8
    q_f32 = rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32)
    k_f32 = rng.standard_normal((batch, heads, head_dim, seqlen)).astype(np.float32)
    v_f32 = rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32)
    feeds = {
        "Q": float32_to_bfloat16_bits(q_f32),
        "K": float32_to_bfloat16_bits(k_f32),
        "V": float32_to_bfloat16_bits(v_f32),
    }
    scale = float32_to_bfloat16_bits(np.array(0.25, dtype=np.float32))
    temperature = float32_to_bfloat16_bits(np.array(2.0, dtype=np.float32))
    zero_mask = float32_to_bfloat16_bits(
        np.zeros((batch, heads, seqlen, seqlen), dtype=np.float32)
    )
    model = _build_bfloat16_mhta_model(feeds, scale, temperature, zero_mask)

    outputs = run_with_iobinding(
        model,
        feeds,
        {name: TensorProto.BFLOAT16 for name in feeds},
        [("Y", TensorProto.BFLOAT16, (batch, heads, seqlen, head_dim))],
        use_musa=True,
    )
    expected = _reference_sdpa(
        bfloat16_bits_to_float32(feeds["Q"]),
        bfloat16_bits_to_float32(feeds["K"]),
        bfloat16_bits_to_float32(feeds["V"]),
        float(bfloat16_bits_to_float32(scale)),
        float(bfloat16_bits_to_float32(temperature)),
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(outputs[0]), expected, rtol=6e-2, atol=6e-2
    )


def test_mhta_scaled_dot_product_attention_sim_rank3_fusion(tmp_path):
    rng = np.random.default_rng(42)
    batch, seqlen, heads, head_dim = 1, 7, 2, 3
    feeds = {
        "K": rng.standard_normal((batch, seqlen, heads, head_dim)).astype(np.float32),
        "Q": rng.standard_normal((batch, 1, heads, head_dim)).astype(np.float32),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32),
        "mask": rng.uniform(-0.25, 0.1, (batch, 1, seqlen)).astype(np.float32),
    }
    scale = np.array(0.1767767, dtype=np.float32)
    temperature_recip = np.array(10.0, dtype=np.float32)
    axes = np.array([2], dtype=np.int64)
    output_shape = np.array([-1, 1, heads * head_dim], dtype=np.int64)

    model = build_graph_model(
        [
            helper.make_node("Einsum", ["K", "Q"], ["Score"], equation="ilhw,bjhw->bhl"),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Add", ["Scaled", "mask"], ["Masked"]),
            helper.make_node("Mul", ["Masked", "temperature_recip"], ["TempScaled"]),
            helper.make_node("Softmax", ["TempScaled"], ["Prob"], axis=-1),
            helper.make_node("Unsqueeze", ["Prob", "axes"], ["Prob4D"]),
            helper.make_node("MatMul", ["Prob4D", "V"], ["Context4D"]),
            helper.make_node("Reshape", ["Context4D", "output_shape"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(temperature_recip, name="temperature_recip"),
            numpy_helper.from_array(axes, name="axes"),
            numpy_helper.from_array(output_shape, name="output_shape"),
        ],
        name="mhta_scaled_dot_product_attention_sim_rank3_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    _, events = _profile_musa_session(model, feeds, tmp_path, "mhta_sdpa_sim_rank3_fusion")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}

    assert fused_ops
    assert "Einsum" not in musa_ops
    assert "Softmax" not in musa_ops
    assert "Reshape" not in musa_ops


def test_mhta_scaled_dot_product_attention_sim_rank3_bl_equation_fusion(tmp_path):
    """TopK-selected branches use an equivalent ``blhw`` label."""
    rng = np.random.default_rng(44)
    batch, seqlen, heads, head_dim = 1, 8, 2, 3
    feeds = {
        "K": rng.standard_normal((batch, seqlen, heads, head_dim)).astype(
            np.float32
        ),
        "Q": rng.standard_normal((batch, 1, heads, head_dim)).astype(np.float32),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(
            np.float32
        ),
        "mask": rng.uniform(-0.4, 0.2, (batch, 1, seqlen)).astype(np.float32),
    }
    scale = np.array(0.2, dtype=np.float32)
    temperature_recip = np.array(7.0, dtype=np.float32)
    axes = np.array([2], dtype=np.int64)
    output_shape = np.array([-1, 1, heads * head_dim], dtype=np.int64)

    model = build_graph_model(
        [
            helper.make_node(
                "Einsum", ["K", "Q"], ["Score"], equation="blhw,bjhw->bhl"
            ),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Add", ["Scaled", "mask"], ["Masked"]),
            helper.make_node(
                "Mul", ["Masked", "temperature_recip"], ["TempScaled"]
            ),
            helper.make_node("Softmax", ["TempScaled"], ["Prob"], axis=-1),
            helper.make_node("Unsqueeze", ["Prob", "axes"], ["Prob4D"]),
            helper.make_node("MatMul", ["Prob4D", "V"], ["Context4D"]),
            helper.make_node("Reshape", ["Context4D", "output_shape"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(temperature_recip, name="temperature_recip"),
            numpy_helper.from_array(axes, name="axes"),
            numpy_helper.from_array(output_shape, name="output_shape"),
        ],
        name="mhta_scaled_dot_product_attention_sim_rank3_bl_equation_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    _, events = _profile_musa_session(
        model, feeds, tmp_path, "mhta_sdpa_sim_rank3_bl_equation"
    )
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {
        op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")
    }

    assert fused_ops
    assert "Einsum" not in musa_ops
    assert "Softmax" not in musa_ops
    assert "Reshape" not in musa_ops


def test_mhta_scaled_dot_product_attention_sim_rank3_div_temperature_fusion(
    tmp_path,
):
    """Keep the reciprocal-Mul regression above and cover Div export form."""
    rng = np.random.default_rng(43)
    batch, seqlen, heads, head_dim = 1, 7, 2, 3
    feeds = {
        "K": rng.standard_normal((batch, seqlen, heads, head_dim)).astype(
            np.float32
        ),
        "Q": rng.standard_normal((batch, 1, heads, head_dim)).astype(np.float32),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(
            np.float32
        ),
        "mask": rng.uniform(-0.3, 0.2, (batch, 1, seqlen)).astype(np.float32),
    }
    scale = np.array(0.25, dtype=np.float32)
    temperature = np.array(0.1, dtype=np.float32)
    axes = np.array([2], dtype=np.int64)
    output_shape = np.array([-1, 1, heads * head_dim], dtype=np.int64)

    model = build_graph_model(
        [
            helper.make_node(
                "Einsum", ["K", "Q"], ["Score"], equation="ilhw,bjhw->bhl"
            ),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Add", ["Scaled", "mask"], ["Masked"]),
            helper.make_node("Div", ["Masked", "temperature"], ["TempScaled"]),
            helper.make_node("Softmax", ["TempScaled"], ["Prob"], axis=-1),
            helper.make_node("Unsqueeze", ["Prob", "axes"], ["Prob4D"]),
            helper.make_node("MatMul", ["Prob4D", "V"], ["Context4D"]),
            helper.make_node("Reshape", ["Context4D", "output_shape"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(temperature, name="temperature"),
            numpy_helper.from_array(axes, name="axes"),
            numpy_helper.from_array(output_shape, name="output_shape"),
        ],
        name="mhta_scaled_dot_product_attention_sim_rank3_div_temperature_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    _, events = _profile_musa_session(
        model, feeds, tmp_path, "mhta_sdpa_sim_rank3_div"
    )
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {
        op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")
    }

    assert fused_ops
    assert "Einsum" not in musa_ops
    assert "Div" not in musa_ops
    assert "Softmax" not in musa_ops
