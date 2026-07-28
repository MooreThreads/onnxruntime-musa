# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""End-to-end tests for the TargetIdCountEmbedding fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_model(dtype: np.dtype, *, detached_count=False) -> bytes:
    dtype = np.dtype(dtype)
    elem_type = helper.np_dtype_to_tensor_dtype(dtype)
    axes_one = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_one")
    axes_two = numpy_helper.from_array(np.array([2], dtype=np.int64), "axes_two")
    reduce_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), "reduce_axes"
    )
    pad = numpy_helper.from_array(np.array(0, dtype=dtype), "pad")
    zero = numpy_helper.from_array(np.array(0, dtype=dtype), "zero")
    cap = numpy_helper.from_array(np.array(3, dtype=dtype), "cap")
    table = numpy_helper.from_array(
        np.array(
            [
                [0.0, 0.25],
                [1.0, 1.25],
                [2.0, 2.25],
                [3.0, 3.25],
            ],
            dtype=np.float32,
        ),
        "table",
    )

    count_source = "ReduceCountDetached" if detached_count else "Count"
    nodes = [
        helper.make_node("Unsqueeze", ["Ids", "axes_two"], ["IdsExpanded"]),
        helper.make_node("Sub", ["IdsExpanded", "Target"], ["Delta"]),
        helper.make_node("Equal", ["Delta", "zero"], ["Hit"]),
        helper.make_node("Equal", ["Ids", "pad"], ["IsPad"]),
        helper.make_node("Not", ["IsPad"], ["IsValid"]),
        helper.make_node("Unsqueeze", ["IsValid", "axes_two"], ["ValidExpanded"]),
        helper.make_node("And", ["Hit", "ValidExpanded"], ["MaskedHit"]),
        helper.make_node("Cast", ["MaskedHit"], ["MaskedHitInt"], to=elem_type),
        helper.make_node(
            "ReduceSum",
            ["MaskedHitInt", "reduce_axes"],
            ["Count"],
            keepdims=0,
        ),
    ]
    if detached_count:
        nodes.append(
            helper.make_node("Identity", ["Count"], ["ReduceCountDetached"])
        )
    nodes.extend(
        [
            helper.make_node("Min", ["Count", "cap"], ["Bucket"]),
            helper.make_node("Unsqueeze", ["Bucket", "axes_one"], ["BucketIndex"]),
            helper.make_node("Gather", ["table", "BucketIndex"], ["Embedding"]),
            helper.make_node(
                "Unsqueeze", [count_source, "axes_one"], ["CountExpanded"]
            ),
            helper.make_node(
                "Cast", ["CountExpanded"], ["CountFeature"], to=TensorProto.FLOAT
            ),
        ]
    )
    for index, node in enumerate(nodes):
        node.name = f"node_{index}"

    graph = helper.make_graph(
        nodes,
        "target_id_count_embedding_fusion",
        [
            helper.make_tensor_value_info("Ids", elem_type, ["B", "S"]),
            helper.make_tensor_value_info("Target", elem_type, []),
        ],
        [
            helper.make_tensor_value_info("Embedding", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("CountFeature", TensorProto.FLOAT, None),
        ],
        initializer=[axes_one, axes_two, reduce_axes, pad, zero, cap, table],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _expected(ids: np.ndarray, target: np.ndarray):
    table = np.array(
        [[0.0, 0.25], [1.0, 1.25], [2.0, 2.25], [3.0, 3.25]],
        dtype=np.float32,
    )
    counts = np.sum((ids != 0) & (ids == target.item()), axis=1).astype(np.int64)
    buckets = np.minimum(counts, 3)
    embedding = table[buckets].reshape(ids.shape[0], 1, 1, 2)
    count_feature = counts.astype(np.float32).reshape(ids.shape[0], 1, 1)
    return embedding, count_feature


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    "ids,target",
    [
        (np.array([[1, 2, 1, 0], [0, 0, 0, 0]], dtype=np.int64), 1),
        (np.array([[5, 5, 5, 5], [5, 0, 5, 7]], dtype=np.int64), 5),
        (np.array([[9, 8, 7], [6, 5, 4]], dtype=np.int64), 1),
    ],
)
def test_target_id_count_embedding_accuracy(dtype, ids, target):
    ids = ids.astype(dtype)
    target = np.array(target, dtype=dtype)
    model = _build_model(dtype)
    actual = run_model_and_compare(
        model, {"Ids": ids, "Target": target}, rtol=0, atol=0
    )
    expected = _expected(ids, target)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=0)


def test_target_id_count_embedding_fusion_assignment(tmp_path):
    ids = np.array([[1, 2, 1, 0], [1, 1, 1, 1]], dtype=np.int64)
    target = np.array(1, dtype=np.int64)
    model = _build_model(np.int64)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "target_id_count_embedding")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    actual = session.run(None, {"Ids": ids, "Target": target})
    expected = _expected(ids, target)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=0)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)

    node_events = [
        event
        for event in events
        if event.get("cat") == "Node"
        and event.get("name", "").endswith("_kernel_time")
    ]
    op_names = {event.get("args", {}).get("op_name") for event in node_events}
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"And", "ReduceSum", "Min", "Gather"} & op_names)


def test_target_id_count_embedding_rejects_detached_count():
    ids = np.array([[1, 2, 1, 0], [1, 1, 1, 1]], dtype=np.int64)
    target = np.array(1, dtype=np.int64)
    model = _build_model(np.int64, detached_count=True)
    actual = run_model_and_compare(
        model, {"Ids": ids, "Target": target}, rtol=0, atol=0
    )
    expected = _expected(ids, target)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=0)
