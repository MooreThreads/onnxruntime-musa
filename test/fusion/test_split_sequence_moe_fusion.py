# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""End-to-end coverage for compact SplitSequenceMoE fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from op_test_utils import musa_devices, run_model_and_compare


def _build_model(split_sizes):
    rng = np.random.default_rng(20260731 + len(split_sizes))
    group_count = len(split_sizes)
    rows = sum(split_sizes)
    width, hidden = 13, 17
    x = (rng.standard_normal((rows, width)) / 3).astype(np.float32)

    nodes = []
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "steps"),
    ]
    graph_inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, ["rows", width])
    ]
    feeds = {"X": x}
    boundaries = [f"boundary_{group}" for group in range(group_count + 1)]
    offset = 0
    for group, boundary in enumerate(boundaries):
        graph_inputs.append(
            helper.make_tensor_value_info(boundary, TensorProto.INT64, [1])
        )
        feeds[boundary] = np.array([offset], dtype=np.int64)
        if group < group_count:
            offset += split_sizes[group]

    starts = boundaries[:-1]
    ends = boundaries[1:]

    ffn_outputs = []
    residual_slices = []
    layer_norm_outputs = []
    for group, size in enumerate(split_sizes):
        w1 = (rng.standard_normal((hidden, width)) / 4).astype(np.float32)
        b1 = (rng.standard_normal(hidden) / 8).astype(np.float32)
        w2 = (rng.standard_normal((width, hidden)) / 4).astype(np.float32)
        b2 = (rng.standard_normal(width) / 8).astype(np.float32)
        gamma = (1.0 + rng.standard_normal(width) / 20).astype(np.float32)
        beta = (rng.standard_normal(width) / 20).astype(np.float32)
        initializers.extend(
            [
                numpy_helper.from_array(w1, f"W1_{group}"),
                numpy_helper.from_array(b1, f"B1_{group}"),
                numpy_helper.from_array(w2, f"W2_{group}"),
                numpy_helper.from_array(b2, f"B2_{group}"),
                numpy_helper.from_array(gamma, f"Gamma_{group}"),
                numpy_helper.from_array(beta, f"Beta_{group}"),
            ]
        )
        pre = f"PreSlice_{group}"
        up = f"Up_{group}"
        relu = f"Relu_{group}"
        down = f"Down_{group}"
        ffn_outputs.append(down)
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    ["X", starts[group], ends[group], "axes", "steps"],
                    [pre],
                    name=f"FFN/{group}/Slice",
                ),
                helper.make_node(
                    "Gemm",
                    [pre, f"W1_{group}", f"B1_{group}"],
                    [up],
                    name=f"FFN/{group}/UpGemm",
                    transB=1,
                ),
                helper.make_node("Relu", [up], [relu], name=f"FFN/{group}/Relu"),
                helper.make_node(
                    "Gemm",
                    [relu, f"W2_{group}", f"B2_{group}"],
                    [down],
                    name=f"FFN/{group}/DownGemm",
                    transB=1,
                ),
            ]
        )

    nodes.extend(
        [
            helper.make_node(
                "Concat", ffn_outputs, ["Ffn"], name="FFN/Concat", axis=0
            ),
            helper.make_node("Add", ["X", "Ffn"], ["Residual"], name="FFN/Add"),
        ]
    )
    for group, size in enumerate(split_sizes):
        sliced = f"ResidualSlice_{group}"
        ln = f"LayerNorm_{group}"
        residual_slices.append(sliced)
        layer_norm_outputs.append(ln)
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    ["Residual", starts[group], ends[group], "axes", "steps"],
                    [sliced],
                    name=f"Norm/{group}/Slice",
                ),
                helper.make_node(
                    "LayerNormalization",
                    [sliced, f"Gamma_{group}", f"Beta_{group}"],
                    [ln],
                    name=f"Norm/{group}/LayerNormalization",
                    axis=-1,
                    epsilon=1.0e-8,
                ),
            ]
        )
    nodes.append(
        helper.make_node(
            "Concat", layer_norm_outputs, ["Y"], name="Norm/Concat", axis=0
        )
    )

    graph = helper.make_graph(
        nodes,
        "split_sequence_moe_compact_boundary",
        graph_inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["rows", width])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


def _profile_node_names(model, feeds, tmp_path):
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "split_sequence_moe")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=options)
    session.run(None, feeds)
    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


@pytest.mark.parametrize("split_sizes", [[2, 3], [1, 0, 3, 2]])
def test_split_sequence_moe_compact_boundary(split_sizes, tmp_path, capfd):
    model, feeds = _build_model(split_sizes)
    run_model_and_compare(model, feeds, rtol=3e-3, atol=3e-3)
    node_names = _profile_node_names(model, feeds, tmp_path)
    fused = [name for name in node_names if name.startswith("MUSAExecutionProvider_")]
    assert len(fused) == 1
    assert not any(
        name.startswith(("Gemm_", "Relu_", "Slice_", "LayerNormalization_", "Concat_"))
        for name in node_names
    )
    captured = capfd.readouterr()
    assert "Unable to get producer node for OrtValueInfo" not in (
        captured.out + captured.err
    )
