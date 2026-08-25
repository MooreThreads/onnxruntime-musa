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


def _profile_musa_session_iobinding(
    model: bytes,
    feeds: dict[str, np.ndarray],
    feed_types: dict[str, int],
    outputs: list[tuple[str, int, tuple[int, ...]]],
    tmp_path,
    prefix: str,
):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    io_binding = session.io_binding()
    for name, arr in feeds.items():
        elem_type = feed_types.get(name, helper.np_dtype_to_tensor_dtype(arr.dtype))
        io_binding.bind_input(name, "cpu", 0, elem_type, arr.shape, arr.ctypes.data)

    output_buffers = []
    for name, elem_type, shape in outputs:
        dtype = np.uint16 if elem_type == TensorProto.BFLOAT16 else np.float16
        output = np.empty(shape, dtype=dtype)
        io_binding.bind_output(name, "cpu", 0, elem_type, output.shape, output.ctypes.data)
        output_buffers.append(output)

    session.run_with_iobinding(io_binding)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    return output_buffers, events


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


def _mhta_bshd_boundary_nodes():
    """UniRank-style BSHD boundary around a BHSD/BHDS attention core."""
    return [
        helper.make_node("Transpose", ["Q0"], ["Q"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["K0"], ["K"], perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["V0"], ["V"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["Q", "K"], ["Score"]),
        helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
        helper.make_node("Add", ["Scaled", "zero_mask"], ["Masked"]),
        helper.make_node("Div", ["Masked", "temperature"], ["TempScaled"]),
        helper.make_node("Softmax", ["TempScaled"], ["Prob"], axis=-1),
        helper.make_node("MatMul", ["Prob", "V"], ["Y"]),
        helper.make_node("Transpose", ["Y"], ["Out"], perm=[0, 2, 1, 3]),
    ]


def _bf16_initializer(name, values):
    values = np.asarray(values, dtype=np.uint16)
    return helper.make_tensor(
        name,
        TensorProto.BFLOAT16,
        values.shape,
        values.tobytes(),
        raw=True,
    )


def _build_bfloat16_mhta_model(feeds, scale, temperature, zero_mask):
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
            _bf16_initializer("scale", scale),
            _bf16_initializer("temperature", temperature),
            _bf16_initializer("zero_mask", zero_mask),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_bfloat16_mhta_bshd_boundary_model(
    feeds, scale, temperature, zero_mask
):
    input_vis = [
        helper.make_tensor_value_info(name, TensorProto.BFLOAT16, value.shape)
        for name, value in feeds.items()
    ]
    graph = helper.make_graph(
        _mhta_bshd_boundary_nodes(),
        "mhta_scaled_dot_product_attention_bfloat16_bshd_boundary_graph",
        input_vis,
        [
            helper.make_tensor_value_info(
                "Out", TensorProto.BFLOAT16, list(feeds["Q0"].shape)
            )
        ],
        initializer=[
            _bf16_initializer("scale", scale),
            _bf16_initializer("temperature", temperature),
            _bf16_initializer("zero_mask", zero_mask),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_float16_mhta_bshd_boundary_model(feeds, scale, temperature, zero_mask):
    input_vis = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, value.shape)
        for name, value in feeds.items()
    ]
    graph = helper.make_graph(
        _mhta_bshd_boundary_nodes(),
        "mhta_scaled_dot_product_attention_float16_bshd_boundary_graph",
        input_vis,
        [
            helper.make_tensor_value_info(
                "Out", TensorProto.FLOAT16, list(feeds["Q0"].shape)
            )
        ],
        initializer=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(temperature, name="temperature"),
            numpy_helper.from_array(zero_mask, name="zero_mask"),
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


def _reference_boolean_sdpa(q, k, v, mask, scale):
    score = np.matmul(q, k) * scale
    masked_score = np.where(mask, score, -np.inf)
    keep_any = np.any(mask, axis=-1, keepdims=True)
    safe_score = np.where(keep_any, masked_score, 0.0)
    masked_score = safe_score - np.max(safe_score, axis=-1, keepdims=True)
    prob = np.exp(masked_score)
    prob /= np.sum(prob, axis=-1, keepdims=True)
    prob = np.where(mask, prob, 0.0)
    return np.matmul(prob, v)


def _reference_lseq_sdpa(q, k, v, scale):
    score = np.matmul(q, k) * scale
    seqlen_k = k.shape[-1]
    keep_limit = max(seqlen_k - 1, 1)
    mask = (np.arange(seqlen_k) < keep_limit).reshape(1, 1, seqlen_k)
    masked_score = np.where(mask, score, -np.inf)
    masked_score = masked_score - np.max(masked_score, axis=-1, keepdims=True)
    prob = np.exp(masked_score)
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


def test_mhta_boolean_mask_scaled_dot_product_attention_fusion(tmp_path):
    """Fuse ranking-gr's two-Where BOOL keep-mask form, including empty rows."""
    rng = np.random.default_rng(53)
    batch, heads, seqlen, head_dim = 1, 3, 5, 4
    feeds = {
        "Q": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32),
        "K": rng.standard_normal((batch, heads, head_dim, seqlen)).astype(np.float32),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32),
        "Mask": rng.random((batch, 1, seqlen, seqlen)) > 0.35,
    }
    # The second Where defines an all-masked query row as zero, not NaN.
    feeds["Mask"][:, :, -1, :] = False
    feeds["Mask"][:, :, :-1, 0] = True
    scale = np.array(1.0 / np.sqrt(head_dim), dtype=np.float32)
    neg_inf = np.array(-np.inf, dtype=np.float32)
    zero = np.array(0.0, dtype=np.float32)
    model = build_graph_model(
        [
            helper.make_node("MatMul", ["Q", "K"], ["Score"]),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Where", ["Mask", "Scaled", "neg_inf"], ["MaskedScore"]),
            helper.make_node("Softmax", ["MaskedScore"], ["Prob"], axis=-1),
            helper.make_node("Where", ["Mask", "Prob", "zero"], ["MaskedProb"]),
            helper.make_node("MatMul", ["MaskedProb", "V"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(neg_inf, name="neg_inf"),
            numpy_helper.from_array(zero, name="zero"),
        ],
        name="mhta_boolean_mask_scaled_dot_product_attention_fusion_graph",
    )

    outputs = run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(outputs[0][:, :, -1, :], 0.0)
    _, events = _profile_musa_session(model, feeds, tmp_path, "mhta_sdpa_bool_mask")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "Softmax" not in musa_ops
    assert "Where" not in musa_ops
    assert "Mul" not in musa_ops


def test_mhta_absorbs_ranking_gr_int32_mask_preprocessing(tmp_path):
    """Consume the shared INT32 mask before Slice/Cast/Equal."""
    rng = np.random.default_rng(54)
    batch, heads, seqlen, head_dim = 1, 3, 5, 4
    raw_mask = rng.integers(0, 3, (batch, 1, seqlen, seqlen + 2), dtype=np.int32)
    raw_mask[:, :, :-1, 0] = 1
    raw_mask[:, :, -1, :seqlen] = 0
    feeds = {
        "Q": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32),
        "K": rng.standard_normal((batch, heads, head_dim, seqlen)).astype(np.float32),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32),
        "RawMask": raw_mask,
    }
    scale = np.array(1.0 / np.sqrt(head_dim), dtype=np.float32)
    neg_inf = np.array(-np.inf, dtype=np.float32)
    zero = np.array(0.0, dtype=np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([seqlen], dtype=np.int64)
    axes = np.array([3], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    model = build_graph_model(
        [
            helper.make_node("Slice", ["RawMask", "starts", "ends", "axes", "steps"], ["MaskSlice"]),
            helper.make_node("Cast", ["MaskSlice"], ["MaskInt64"], to=TensorProto.INT64),
            helper.make_node("Equal", ["MaskInt64", "one"], ["Mask"]),
            helper.make_node("MatMul", ["Q", "K"], ["Score"]),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Where", ["Mask", "Scaled", "neg_inf"], ["MaskedScore"]),
            helper.make_node("Softmax", ["MaskedScore"], ["Prob"], axis=-1),
            helper.make_node("Where", ["Mask", "Prob", "zero"], ["MaskedProb"]),
            helper.make_node("MatMul", ["MaskedProb", "V"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=[
            numpy_helper.from_array(value, name=name)
            for name, value in {
                "scale": scale,
                "neg_inf": neg_inf,
                "zero": zero,
                "starts": starts,
                "ends": ends,
                "axes": axes,
                "steps": steps,
                "one": one,
            }.items()
        ],
        name="mhta_ranking_gr_raw_int32_mask_graph",
    )

    outputs = run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(outputs[0][:, :, -1, :], 0.0)
    _, events = _profile_musa_session(model, feeds, tmp_path, "mhta_raw_int32_mask")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert not {"Slice", "Cast", "Equal", "Where", "Softmax", "Mul"} & musa_ops


def test_mhta_unirank_lseq_materialized_bool_mask_runmath(tmp_path):
    """Fold UniRank lseq rank-3 pre-mask SDPA and materialize its BOOL keep mask."""
    rng = np.random.default_rng(57)
    folded_heads, seqlen_q, seqlen_k, head_dim = 4, 3, 5, 8
    feeds = {
        "Q": rng.standard_normal((folded_heads, seqlen_q, head_dim)).astype(np.float32),
        "K": rng.standard_normal((folded_heads, head_dim, seqlen_k)).astype(np.float32),
        "V": rng.standard_normal((folded_heads, seqlen_k, head_dim)).astype(np.float32),
    }
    scale = np.array(1.0 / np.sqrt(head_dim), dtype=np.float32)
    neg_inf = np.array(-np.inf, dtype=np.float32)
    zero_i64 = np.array(0, dtype=np.int64)
    one_i64 = np.array(1, dtype=np.int64)
    two_i64 = np.array(2, dtype=np.int64)
    axis0 = np.array([0], dtype=np.int64)
    axis1 = np.array([1], dtype=np.int64)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["Q", "K"], ["Score"]),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Shape", ["Scaled"], ["ScoreShape"]),
            helper.make_node("Gather", ["ScoreShape", "two_i64"], ["Sk"], axis=0),
            helper.make_node("Sub", ["Sk", "one_i64"], ["SkMinusOne"]),
            helper.make_node("Clip", ["SkMinusOne", "one_i64"], ["KeepLimit"]),
            helper.make_node("Range", ["zero_i64", "Sk", "one_i64"], ["KeyIndex"]),
            helper.make_node("Less", ["KeyIndex", "KeepLimit"], ["Keep1D"]),
            helper.make_node("Unsqueeze", ["Keep1D", "axis0"], ["Keep2D"]),
            helper.make_node("Unsqueeze", ["Keep2D", "axis1"], ["Keep3D"]),
            helper.make_node("Where", ["Keep3D", "Scaled", "neg_inf"], ["MaskedScore"]),
            helper.make_node("Softmax", ["MaskedScore"], ["Prob"], axis=-1),
            helper.make_node("MatMul", ["Prob", "V"], ["Y"]),
        ],
        "mhta_unirank_lseq_materialized_bool_mask_runmath_graph",
        [
            helper.make_tensor_value_info("Q", TensorProto.FLOAT, ["BH", "Sq", "D"]),
            helper.make_tensor_value_info("K", TensorProto.FLOAT, ["BH", "D", "Sk"]),
            helper.make_tensor_value_info("V", TensorProto.FLOAT, ["BH", "Sk", "D"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["BH", "Sq", "D"])],
        initializer=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(neg_inf, name="neg_inf"),
            numpy_helper.from_array(zero_i64, name="zero_i64"),
            numpy_helper.from_array(one_i64, name="one_i64"),
            numpy_helper.from_array(two_i64, name="two_i64"),
            numpy_helper.from_array(axis0, name="axis0"),
            numpy_helper.from_array(axis1, name="axis1"),
        ],
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model_proto.ir_version = min(model_proto.ir_version, 10)
    model = model_proto.SerializeToString()

    outputs = run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
    expected = _reference_lseq_sdpa(feeds["Q"], feeds["K"], feeds["V"], float(scale))
    np.testing.assert_allclose(outputs[0], expected, rtol=1e-4, atol=1e-4)
    _, events = _profile_musa_session(model, feeds, tmp_path, "mhta_unirank_lseq_runmath")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert not {
        "Shape",
        "Gather",
        "Sub",
        "Clip",
        "Range",
        "Less",
        "Unsqueeze",
        "Where",
        "Softmax",
        "Mul",
    } & musa_ops


def test_mhta_bfloat16_raw_int32_mask_runflash(tmp_path):
    """BF16 ranking-gr keep-mask topology should fuse and run via RunFlash."""
    rng = np.random.default_rng(55)
    batch, heads, seqlen, head_dim = 1, 2, 4, 8
    q_f32 = rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32)
    k_f32 = rng.standard_normal((batch, heads, head_dim, seqlen)).astype(np.float32)
    v_f32 = rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float32)
    raw_mask = rng.integers(0, 3, (batch, 1, seqlen, seqlen + 1), dtype=np.int32)
    raw_mask[:, :, :, 0] = 1
    raw_mask[:, :, -1, :seqlen] = 0
    feeds = {
        "Q": float32_to_bfloat16_bits(q_f32),
        "K": float32_to_bfloat16_bits(k_f32),
        "V": float32_to_bfloat16_bits(v_f32),
        "RawMask": raw_mask,
    }
    scale = float32_to_bfloat16_bits(
        np.array(1.0 / np.sqrt(head_dim), dtype=np.float32)
    )
    neg_inf = float32_to_bfloat16_bits(np.array(-np.inf, dtype=np.float32))
    zero = float32_to_bfloat16_bits(np.array(0.0, dtype=np.float32))
    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice", ["RawMask", "starts", "ends", "axes", "steps"], ["MaskSlice"]
            ),
            helper.make_node("Cast", ["MaskSlice"], ["MaskInt64"], to=TensorProto.INT64),
            helper.make_node("Equal", ["MaskInt64", "one"], ["Mask"]),
            helper.make_node("MatMul", ["Q", "K"], ["Score"]),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Where", ["Mask", "Scaled", "neg_inf"], ["MaskedScore"]),
            helper.make_node("Softmax", ["MaskedScore"], ["Prob"], axis=-1),
            helper.make_node("Cast", ["Prob"], ["ProbF32"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["ProbF32"], ["ProbTyped"], to=TensorProto.BFLOAT16),
            helper.make_node("Where", ["Mask", "ProbTyped", "zero"], ["MaskedProb"]),
            helper.make_node("MatMul", ["MaskedProb", "V"], ["Y"]),
        ],
        "mhta_bfloat16_raw_int32_mask_fusion_match_graph",
        [
            helper.make_tensor_value_info(
                "Q", TensorProto.BFLOAT16, list(feeds["Q"].shape)
            ),
            helper.make_tensor_value_info(
                "K", TensorProto.BFLOAT16, list(feeds["K"].shape)
            ),
            helper.make_tensor_value_info(
                "V", TensorProto.BFLOAT16, list(feeds["V"].shape)
            ),
            helper.make_tensor_value_info(
                "RawMask", TensorProto.INT32, list(feeds["RawMask"].shape)
            ),
        ],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.BFLOAT16, [batch, heads, seqlen, head_dim]
            )
        ],
        initializer=[
            _bf16_initializer("scale", scale),
            _bf16_initializer("neg_inf", neg_inf),
            _bf16_initializer("zero", zero),
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts"),
            numpy_helper.from_array(np.array([seqlen], dtype=np.int64), name="ends"),
            numpy_helper.from_array(np.array([3], dtype=np.int64), name="axes"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps"),
            numpy_helper.from_array(np.array(1, dtype=np.int64), name="one"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    outputs, events = _profile_musa_session_iobinding(
        model.SerializeToString(),
        feeds,
        {
            "Q": TensorProto.BFLOAT16,
            "K": TensorProto.BFLOAT16,
            "V": TensorProto.BFLOAT16,
        },
        [("Y", TensorProto.BFLOAT16, (batch, heads, seqlen, head_dim))],
        tmp_path,
        "mhta_bfloat16_raw_int32_mask",
    )
    expected = _reference_boolean_sdpa(
        bfloat16_bits_to_float32(feeds["Q"]),
        bfloat16_bits_to_float32(feeds["K"]),
        bfloat16_bits_to_float32(feeds["V"]),
        raw_mask[:, :, :, :seqlen] == 1,
        float(bfloat16_bits_to_float32(scale)),
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(outputs[0]), expected, rtol=6e-2, atol=6e-2
    )
    np.testing.assert_array_equal(bfloat16_bits_to_float32(outputs[0])[:, :, -1, :], 0.0)
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert not {"Slice", "Cast", "Equal", "Where", "Softmax", "Mul"} & musa_ops


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


def test_mhta_boolean_mask_scaled_dot_product_attention_fp16_runflash(tmp_path):
    rng = np.random.default_rng(56)
    batch, heads, seqlen, head_dim = 1, 2, 4, 8
    feeds = {
        "Q": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float16),
        "K": rng.standard_normal((batch, heads, head_dim, seqlen)).astype(np.float16),
        "V": rng.standard_normal((batch, heads, seqlen, head_dim)).astype(np.float16),
        "Mask": rng.random((batch, 1, seqlen, seqlen)) > 0.35,
    }
    feeds["Mask"][:, :, :, 0] = True
    feeds["Mask"][:, :, -1, :] = False
    scale = np.array(1.0 / np.sqrt(head_dim), dtype=np.float16)
    neg_inf = np.array(-np.inf, dtype=np.float16)
    zero = np.array(0.0, dtype=np.float16)
    model = build_graph_model(
        [
            helper.make_node("MatMul", ["Q", "K"], ["Score"]),
            helper.make_node("Mul", ["Score", "scale"], ["Scaled"]),
            helper.make_node("Where", ["Mask", "Scaled", "neg_inf"], ["MaskedScore"]),
            helper.make_node("Softmax", ["MaskedScore"], ["Prob"], axis=-1),
            helper.make_node("Cast", ["Prob"], ["ProbF32"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["ProbF32"], ["ProbTyped"], to=TensorProto.FLOAT16),
            helper.make_node("Where", ["Mask", "ProbTyped", "zero"], ["MaskedProb"]),
            helper.make_node("MatMul", ["MaskedProb", "V"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT16)],
        initializers=[
            numpy_helper.from_array(scale, name="scale"),
            numpy_helper.from_array(neg_inf, name="neg_inf"),
            numpy_helper.from_array(zero, name="zero"),
        ],
        name="mhta_boolean_mask_scaled_dot_product_attention_fp16_graph",
    )

    run_model_and_compare(model, feeds, rtol=3e-2, atol=3e-2)
    outputs, _ = _profile_musa_session(model, feeds, tmp_path, "mhta_sdpa_bool_fp16")
    np.testing.assert_array_equal(outputs[0][:, :, -1, :], 0.0)
    _, events = _profile_musa_session(model, feeds, tmp_path, "mhta_sdpa_bool_fp16_profile")
    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert not {"Where", "Softmax", "Mul"} & musa_ops


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


def test_mhta_scaled_dot_product_attention_bfloat16_bshd_boundary_runflash(tmp_path):
    """Absorb UniRank's BSHD<->BHSD/BHDS boundary transposes into SDPA."""
    rng = np.random.default_rng(47)
    batch, seqlen, heads, head_dim = 1, 4, 2, 8
    q_f32 = rng.standard_normal((batch, seqlen, heads, head_dim)).astype(np.float32)
    k_f32 = rng.standard_normal((batch, seqlen, heads, head_dim)).astype(np.float32)
    v_f32 = rng.standard_normal((batch, seqlen, heads, head_dim)).astype(np.float32)
    feeds = {
        "Q0": float32_to_bfloat16_bits(q_f32),
        "K0": float32_to_bfloat16_bits(k_f32),
        "V0": float32_to_bfloat16_bits(v_f32),
    }
    scale = float32_to_bfloat16_bits(
        np.array(1.0 / np.sqrt(head_dim), dtype=np.float32)
    )
    temperature = float32_to_bfloat16_bits(np.array(2.0, dtype=np.float32))
    zero_mask = float32_to_bfloat16_bits(
        np.zeros((batch, heads, seqlen, seqlen), dtype=np.float32)
    )
    model = _build_bfloat16_mhta_bshd_boundary_model(
        feeds, scale, temperature, zero_mask
    )

    outputs, events = _profile_musa_session_iobinding(
        model,
        feeds,
        {name: TensorProto.BFLOAT16 for name in feeds},
        [("Out", TensorProto.BFLOAT16, (batch, seqlen, heads, head_dim))],
        tmp_path,
        "mhta_bfloat16_bshd_boundary",
    )
    q = bfloat16_bits_to_float32(feeds["Q0"])
    k = bfloat16_bits_to_float32(feeds["K0"])
    v = bfloat16_bits_to_float32(feeds["V0"])
    expected_logical = _reference_sdpa(
        np.transpose(q, (0, 2, 1, 3)),
        np.transpose(k, (0, 2, 3, 1)),
        np.transpose(v, (0, 2, 1, 3)),
        float(bfloat16_bits_to_float32(scale)),
        float(bfloat16_bits_to_float32(temperature)),
    )
    expected = np.transpose(expected_logical, (0, 2, 1, 3))
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(outputs[0]), expected, rtol=6e-2, atol=6e-2
    )

    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}
    assert fused_ops
    assert "Transpose" not in musa_ops


def test_mhta_scaled_dot_product_attention_float16_bshd_boundary_runflash(tmp_path):
    """FP16 coverage for the UniRank BSHD boundary strided RunFlash path."""
    rng = np.random.default_rng(48)
    batch, seqlen, heads, head_dim = 1, 4, 2, 8
    feeds = {
        "Q0": rng.standard_normal((batch, seqlen, heads, head_dim)).astype(np.float16),
        "K0": rng.standard_normal((batch, seqlen, heads, head_dim)).astype(np.float16),
        "V0": rng.standard_normal((batch, seqlen, heads, head_dim)).astype(np.float16),
    }
    scale = np.array(1.0 / np.sqrt(head_dim), dtype=np.float16)
    temperature = np.array(2.0, dtype=np.float16)
    zero_mask = np.zeros((batch, heads, seqlen, seqlen), dtype=np.float16)
    model = _build_float16_mhta_bshd_boundary_model(
        feeds, scale, temperature, zero_mask
    )

    outputs, events = _profile_musa_session_iobinding(
        model,
        feeds,
        {name: TensorProto.FLOAT16 for name in feeds},
        [("Out", TensorProto.FLOAT16, (batch, seqlen, heads, head_dim))],
        tmp_path,
        "mhta_float16_bshd_boundary",
    )
    expected_logical = _reference_sdpa(
        np.transpose(feeds["Q0"].astype(np.float32), (0, 2, 1, 3)),
        np.transpose(feeds["K0"].astype(np.float32), (0, 2, 3, 1)),
        np.transpose(feeds["V0"].astype(np.float32), (0, 2, 1, 3)),
        float(scale.astype(np.float32)),
        float(temperature.astype(np.float32)),
    )
    expected = np.transpose(expected_logical, (0, 2, 1, 3))
    np.testing.assert_allclose(
        outputs[0].astype(np.float32), expected, rtol=6e-2, atol=6e-2
    )

    musa_ops = _ops_by_provider(events).get("MUSAExecutionProvider", set())
    fused_ops = {op for op in musa_ops if str(op).startswith("MUSAExecutionProvider_")}
    assert fused_ops
    assert "Transpose" not in musa_ops


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
