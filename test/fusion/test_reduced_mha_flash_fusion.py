"""End-to-end coverage for Unsqueeze -> Attention -> Reshape -> Gemm fusion."""

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices


def _reference(x, qkv_weight, qkv_bias, mask, out_weight, out_bias, heads, scale):
    sequence, _ = x.shape
    attention_dim = qkv_weight.shape[1] // 3
    head_dim = attention_dim // heads
    qkv = x @ qkv_weight + qkv_bias
    q, k, v = np.split(qkv, 3, axis=1)
    q = q.reshape(sequence, heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(sequence, heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(sequence, heads, head_dim).transpose(1, 0, 2)
    score = np.einsum("hsd,htd->hst", q, k) * scale
    score = np.where(mask[0] != 0, score, -np.inf)
    prob = np.exp(score - np.max(score, axis=-1, keepdims=True))
    prob /= np.sum(prob, axis=-1, keepdims=True)
    attention = np.einsum("hst,htd->hsd", prob, v).transpose(1, 0, 2)
    return attention.reshape(sequence, attention_dim) @ out_weight.T + out_bias


def _profile(model, feeds, tmp_path):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "reduced_mha_flash")
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    result = session.run(None, feeds)
    profile = session.end_profiling()
    try:
        with open(profile, encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile):
            os.remove(profile)
    return result, events


def test_reduced_mha_flash_fuses_four_attention_nodes(tmp_path):
    rng = np.random.default_rng(20260730)
    sequence, input_dim, attention_dim, output_dim, heads = 5, 12, 8, 10, 2
    x = rng.standard_normal((sequence, input_dim)).astype(np.float32)
    qkv_weight = rng.standard_normal((input_dim, 3 * attention_dim)).astype(np.float32) / 5
    qkv_bias = rng.standard_normal(3 * attention_dim).astype(np.float32) / 9
    out_weight = rng.standard_normal((output_dim, attention_dim)).astype(np.float32) / 7
    out_bias = rng.standard_normal(output_dim).astype(np.float32) / 11
    mask = np.ones((1, 1, sequence, sequence), dtype=np.int32)
    mask[:, :, :, -1] = 0
    scale = 0.5

    graph = helper.make_graph(
        [
            helper.make_node("Unsqueeze", ["X", "axes"], ["X3"]),
            helper.make_node(
                "Attention", ["X3", "qkv_weight", "qkv_bias", "mask"], ["A"],
                domain="com.microsoft", num_heads=heads,
                qkv_hidden_sizes=[attention_dim] * 3, scale=scale,
            ),
            helper.make_node("Reshape", ["A", "shape"], ["A2"]),
            helper.make_node("Gemm", ["A2", "out_weight", "out_bias"], ["Y"], transB=1),
        ],
        "reduced_mha_flash_four_node_graph",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape),
            helper.make_tensor_value_info("mask", TensorProto.INT32, mask.shape),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes"),
            numpy_helper.from_array(np.array([-1, attention_dim], dtype=np.int64), name="shape"),
            numpy_helper.from_array(qkv_weight, name="qkv_weight"),
            numpy_helper.from_array(qkv_bias, name="qkv_bias"),
            numpy_helper.from_array(out_weight, name="out_weight"),
            numpy_helper.from_array(out_bias, name="out_bias"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = min(model.ir_version, 10)
    actual, events = _profile(model.SerializeToString(), {"X": x, "mask": mask}, tmp_path)
    expected = _reference(x, qkv_weight, qkv_bias, mask, out_weight, out_bias, heads, scale)
    np.testing.assert_allclose(actual[0], expected, rtol=2e-4, atol=2e-4)

    musa_ops = {
        event.get("args", {}).get("op_name")
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "MUSAExecutionProvider"
    }
    assert any(str(name).startswith("MUSAExecutionProvider_") for name in musa_ops)
    assert "Attention" not in musa_ops
    assert "Gemm" not in musa_ops
