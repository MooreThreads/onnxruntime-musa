# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the Plugin EP Concat -> MatMul fusion pattern."""

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


def test_concat_matmul_fusion_concat_on_lhs():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    x1 = rng.standard_normal((2, 3, 4, 7)).astype(np.float32)
    b = rng.standard_normal((2, 3, 12, 6)).astype(np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=-1),
        helper.make_node("MatMul", ["C", "B"], ["Y"]),
    ]
    feeds = {"X0": x0, "X1": x1, "B": b}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="concat_matmul_lhs_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_concat_matmul_fusion_concat_on_rhs():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 3, 4, 6)).astype(np.float32)
    x0 = rng.standard_normal((2, 3, 6, 5)).astype(np.float32)
    x1 = rng.standard_normal((2, 3, 6, 7)).astype(np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=-1),
        helper.make_node("MatMul", ["A", "C"], ["Y"]),
    ]
    feeds = {"A": a, "X0": x0, "X1": x1}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="concat_matmul_rhs_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)


def test_dynamic_concat_matmul_rank_broadcast_not_fused():
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((5, 3, 4)).astype(np.float32)
    x1 = rng.standard_normal((5, 3, 5)).astype(np.float32)
    b = rng.standard_normal((9, 6)).astype(np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=-1),
        helper.make_node("MatMul", ["C", "B"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dynamic_concat_matmul_rank_broadcast_graph",
        [
            helper.make_tensor_value_info(
                "X0", TensorProto.FLOAT, ["batch", 3, 4]
            ),
            helper.make_tensor_value_info(
                "X1", TensorProto.FLOAT, ["batch", 3, 5]
            ),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [9, 6]),
        ],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, ["batch", 3, 6]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)

    run_model_and_compare(
        model.SerializeToString(),
        {"X0": x0, "X1": x1, "B": b},
        rtol=1e-3,
        atol=1e-3,
    )


def test_dynamic_concat_matmul_equal_rank_fuses(tmp_path):
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    x1 = rng.standard_normal((2, 3, 4, 7)).astype(np.float32)
    b = rng.standard_normal((2, 3, 12, 6)).astype(np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=-1),
        helper.make_node("MatMul", ["C", "B"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dynamic_concat_matmul_equal_rank_fusion_graph",
        [
            helper.make_tensor_value_info("X0", TensorProto.FLOAT, ["batch", 3, 4, 5]),
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, ["batch", 3, 4, 7]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, ["batch", 3, 12, 6]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 3, 4, 6])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    model_bytes = model.SerializeToString()
    feeds = {"X0": x0, "X1": x1, "B": b}

    run_model_and_compare(model_bytes, feeds, rtol=1e-3, atol=1e-3)
    musa_ops = _profile_musa_ops(
        model_bytes, feeds, tmp_path, "dynamic_concat_matmul_equal_rank"
    )
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "Concat" not in musa_ops
    assert "MatMul" not in musa_ops


def test_concat_matmul_fusion_allows_empty_output(tmp_path):
    x0 = np.empty((1, 0, 2), dtype=np.float32)
    x1 = np.empty((1, 0, 2), dtype=np.float32)
    b = np.empty((2, 2, 0), dtype=np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=0),
        helper.make_node("MatMul", ["C", "B"], ["Y"]),
    ]
    feeds = {"X0": x0, "X1": x1, "B": b}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="concat_matmul_empty_output_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    musa_ops = _profile_musa_ops(model, feeds, tmp_path, "concat_matmul_empty_output")
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)


def test_concat_matmul_fusion_zero_k_non_empty_output(tmp_path):
    x0 = np.empty((1, 3, 0), dtype=np.float32)
    x1 = np.empty((1, 2, 0), dtype=np.float32)
    b = np.empty((1, 0, 4), dtype=np.float32)

    nodes = [
        helper.make_node("Concat", ["X0", "X1"], ["C"], axis=1),
        helper.make_node("MatMul", ["C", "B"], ["Y"]),
    ]
    feeds = {"X0": x0, "X1": x1, "B": b}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y", TensorProto.FLOAT)],
        name="concat_matmul_zero_k_fusion_graph",
    )

    outputs = run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(outputs[0], np.zeros((1, 5, 4), dtype=np.float32))
    musa_ops = _profile_musa_ops(model, feeds, tmp_path, "concat_matmul_zero_k")
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
