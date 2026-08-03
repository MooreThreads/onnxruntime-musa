# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""End-to-end tests for the RecRankCalibration fusion."""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from op_test_utils import musa_devices, run_model_and_compare


def _initializer(name: str, value: np.ndarray):
    return numpy_helper.from_array(value, name=name)


def _make_calibration_branch(
    prefix: str, bucket_size: int, tile: str, complement_from_raw: bool
):
    bucket = f"{prefix}/bucket"
    inverse_bucket = f"{prefix}/inverse_bucket"
    epsilon = f"{prefix}/epsilon"
    one = f"{prefix}/one"
    axes = f"{prefix}/axes"
    task_relu = f"{prefix}/task_relu"
    task_scores = f"{prefix}/task_scores"
    output = f"{prefix}/output"

    nodes = [
        helper.make_node(
            "Mul", [task_scores, bucket], [f"{prefix}/scaled"], name=f"{prefix}/Mul"
        ),
        helper.make_node(
            "Floor", [f"{prefix}/scaled"], [f"{prefix}/left"], name=f"{prefix}/Floor"
        ),
        helper.make_node(
            "Less", [tile, f"{prefix}/left"], [f"{prefix}/less"], name=f"{prefix}/Less"
        ),
        helper.make_node(
            "Cast",
            [f"{prefix}/less"],
            [f"{prefix}/less_double"],
            name=f"{prefix}/CastLess",
            to=TensorProto.DOUBLE,
        ),
        helper.make_node(
            "Equal",
            [tile, f"{prefix}/left"],
            [f"{prefix}/equal"],
            name=f"{prefix}/Equal",
        ),
        helper.make_node(
            "Cast",
            [f"{prefix}/equal"],
            [f"{prefix}/equal_double"],
            name=f"{prefix}/CastEqual",
            to=TensorProto.DOUBLE,
        ),
        helper.make_node(
            "Sub",
            [f"{prefix}/scaled", f"{prefix}/left"],
            [f"{prefix}/frac"],
            name=f"{prefix}/SubFrac",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}/equal_double", f"{prefix}/frac"],
            [f"{prefix}/fractional_weight"],
            name=f"{prefix}/MulFrac",
        ),
        helper.make_node(
            "Add",
            [f"{prefix}/less_double", f"{prefix}/fractional_weight"],
            [f"{prefix}/mask"],
            name=f"{prefix}/AddMask",
        ),
        helper.make_node(
            "Mul",
            [task_relu, f"{prefix}/mask"],
            [f"{prefix}/weighted"],
            name=f"{prefix}/MulWeighted",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}/weighted", inverse_bucket],
            [f"{prefix}/normalized"],
            name=f"{prefix}/MulNormalize",
        ),
        helper.make_node(
            "ReduceSum",
            [f"{prefix}/normalized", axes],
            [f"{prefix}/probability_double"],
            name=f"{prefix}/ReduceSum",
            keepdims=1,
        ),
        helper.make_node(
            "Cast",
            [f"{prefix}/probability_double"],
            [f"{prefix}/probability_float"],
            name=f"{prefix}/CastOutput",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Clip",
            [f"{prefix}/probability_float", epsilon, one],
            [f"{prefix}/probability"],
            name=f"{prefix}/ClipProbability",
        ),
        helper.make_node(
            "Sub",
            [
                one,
                (
                    f"{prefix}/probability_float"
                    if complement_from_raw
                    else f"{prefix}/probability"
                ),
            ],
            [f"{prefix}/complement_raw"],
            name=f"{prefix}/SubComplement",
        ),
        helper.make_node(
            "Clip",
            [f"{prefix}/complement_raw", epsilon, one],
            [f"{prefix}/complement"],
            name=f"{prefix}/ClipComplement",
        ),
        helper.make_node(
            "Div",
            [f"{prefix}/probability", f"{prefix}/complement"],
            [f"{prefix}/odds"],
            name=f"{prefix}/Div",
        ),
        helper.make_node(
            "Log", [f"{prefix}/odds"], [output], name=f"{prefix}/Log"
        ),
    ]
    initializers = [
        _initializer(bucket, np.array(bucket_size, dtype=np.float64)),
        _initializer(
            inverse_bucket, np.array(1.0 / bucket_size, dtype=np.float64)
        ),
        _initializer(epsilon, np.array(1.0e-16, dtype=np.float32)),
        _initializer(one, np.array(1.0, dtype=np.float32)),
        _initializer(axes, np.array([-1], dtype=np.int64)),
    ]
    return nodes, initializers, task_relu, task_scores, output


def _build_model(
    branch_bucket_sizes: list[int],
    prefix_shape: tuple[int, ...],
    raw_complement_indices: set[int] | None = None,
):
    nodes = []
    initializers = []
    graph_inputs = []
    graph_outputs = []
    feeds = {}
    tile_names = {}

    for bucket_size in sorted(set(branch_bucket_sizes)):
        base_name = f"positions_{bucket_size}"
        repeats_name = f"repeats_{bucket_size}"
        tile_name = f"tile_{bucket_size}"
        tile_names[bucket_size] = tile_name
        graph_inputs.append(
            helper.make_tensor_value_info(
                base_name, TensorProto.DOUBLE, [1] * len(prefix_shape) + [bucket_size]
            )
        )
        feeds[base_name] = np.arange(bucket_size, dtype=np.float64).reshape(
            [1] * len(prefix_shape) + [bucket_size]
        )
        initializers.append(
            _initializer(
                repeats_name, np.array([*prefix_shape, 1], dtype=np.int64)
            )
        )
        nodes.append(
            helper.make_node(
                "Tile", [base_name, repeats_name], [tile_name], name=f"Tile{bucket_size}"
            )
        )

    rng = np.random.default_rng(20260803)
    for index, bucket_size in enumerate(branch_bucket_sizes):
        prefix = f"calibration_{index}"
        branch_nodes, branch_initializers, task_relu, task_scores, output = (
            _make_calibration_branch(
                prefix,
                bucket_size,
                tile_names[bucket_size],
                index in (raw_complement_indices or set()),
            )
        )
        nodes.extend(branch_nodes)
        initializers.extend(branch_initializers)
        relu_shape = [*prefix_shape, bucket_size]
        score_shape = [*prefix_shape, 1]
        graph_inputs.extend(
            [
                helper.make_tensor_value_info(
                    task_relu, TensorProto.DOUBLE, relu_shape
                ),
                helper.make_tensor_value_info(
                    task_scores, TensorProto.DOUBLE, score_shape
                ),
            ]
        )
        graph_outputs.append(
            helper.make_tensor_value_info(output, TensorProto.FLOAT, score_shape)
        )
        feeds[task_relu] = rng.uniform(0.01, 0.99, size=relu_shape).astype(
            np.float64
        )
        scores = rng.uniform(0.02, 0.98, size=score_shape).astype(np.float64)
        scores.reshape(-1)[0] = 0.0
        if scores.size > 1:
            scores.reshape(-1)[-1] = 1.0
        feeds[task_scores] = scores

    graph = helper.make_graph(
        nodes,
        "rec_rank_calibration_fusion",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


def _profile_fused_count(model: bytes, feeds: dict[str, np.ndarray], tmp_path) -> int:
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "rec_rank_calibration_fusion")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    return sum(
        1
        for event in events
        if event.get("cat") == "Node"
        and event.get("name", "").endswith("_kernel_time")
        and str(event.get("args", {}).get("op_name", "")).startswith(
            "MUSAExecutionProvider_"
        )
    )


@pytest.mark.parametrize(
    ("bucket_size", "prefix_shape", "complement_from_raw"),
    [(100, (4,), False), (1000, (2, 3), True)],
)
def test_rec_rank_calibration_fusion_bucket_sizes_and_ranks(
    bucket_size, prefix_shape, complement_from_raw, tmp_path
):
    model, feeds = _build_model(
        [bucket_size],
        prefix_shape,
        raw_complement_indices={0} if complement_from_raw else set(),
    )
    run_model_and_compare(model, feeds, rtol=2e-5, atol=2e-5)
    assert _profile_fused_count(model, feeds, tmp_path) == 1


def test_rec_rank_calibration_matches_24_documented_branches(tmp_path):
    # The inspected recommendation graph has 23 bucket-100 branches and one
    # bucket-1000 branch, sharing one Tile producer per bucket size.
    model, feeds = _build_model(
        [100] * 23 + [1000], (2,), raw_complement_indices={21, 22}
    )
    run_model_and_compare(model, feeds, rtol=2e-5, atol=2e-5)
    assert _profile_fused_count(model, feeds, tmp_path) == 24
