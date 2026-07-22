# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the SIM SegmentMaxBroadcast fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_segment_max_broadcast_model(*, detached_final_indices=False) -> bytes:
    axes_minus_one = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name="const_fold_opt__11684"
    )
    axes_zero = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="const_axes__11245"
    )
    axes_one = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="axes_const__9894"
    )
    range_start = numpy_helper.from_array(
        np.array(0, dtype=np.int64),
        name="VocabFileEmbeddingLookup/brow_300_time_list/GreaterEqual/y:0",
    )
    range_delta = numpy_helper.from_array(
        np.array(1, dtype=np.int64), name="add_9/y:0"
    )
    fill_value = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name="value"
    )

    nodes = [
        helper.make_node(
            "Unique",
            ["Reshape:0"],
            ["y__10178", "idx_first__10179", "idx__10180", "counts__10181"],
            name="Unique",
            sorted=0,
        ),
        helper.make_node(
            "Cast",
            ["idx__10180"],
            ["Unique__10183_cast:0"],
            name="Unique__10183_cast",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Cast",
            ["Unique__10183_cast:0"],
            ["Cast__11406:0"],
            name="Cast__11406",
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Shape", ["Cast__11406:0"], ["Shape__11411:0"], name="Shape__11411"
        ),
        helper.make_node(
            "TopK",
            ["Cast__11406:0", "Shape__11411:0"],
            ["UnsortedSegmentMax_TopK__11415:0", "UnsortedSegmentMax_TopK__11415:1"],
            name="UnsortedSegmentMax_TopK__11415",
            axis=0,
            largest=0,
            sorted=1,
        ),
        helper.make_node(
            "Unsqueeze",
            ["UnsortedSegmentMax_TopK__11415:0", "const_fold_opt__11684"],
            ["Unsqueeze__11434:0"],
            name="Unsqueeze__11434",
        ),
        helper.make_node(
            "Unique",
            ["UnsortedSegmentMax_TopK__11415:0"],
            ["UnsortedSegmentMax_Unique__11417:0", "", "UnsortedSegmentMax_Unique__11417:2", "UnsortedSegmentMax_Unique__11417:3"],
            name="UnsortedSegmentMax_Unique__11417",
            axis=0,
            sorted=1,
        ),
        helper.make_node(
            "ReduceMax",
            ["UnsortedSegmentMax_Unique__11417:3", "const_axes__11245"],
            ["ReduceMax__11420:0"],
            name="ReduceMax__11420",
            keepdims=1,
        ),
        helper.make_node(
            "Gather",
            ["UnsortedSegmentMax_Unique__11417:3", "UnsortedSegmentMax_Unique__11417:2"],
            ["Gather__11425:0"],
            name="Gather__11425",
        ),
        helper.make_node(
            "Squeeze",
            ["Shape__11411:0", "const_axes__11245"],
            ["Squeeze__11414:0"],
            name="Squeeze__11414",
        ),
        helper.make_node(
            "Range",
            ["VocabFileEmbeddingLookup/brow_300_time_list/GreaterEqual/y:0", "Squeeze__11414:0", "add_9/y:0"],
            ["Range__11424:0"],
            name="Range__11424",
        ),
        helper.make_node(
            "Mod",
            ["Range__11424:0", "Gather__11425:0"],
            ["Mod__11428:0"],
            name="Mod__11428",
        ),
        helper.make_node(
            "Unsqueeze",
            ["Mod__11428:0", "const_fold_opt__11684"],
            ["Unsqueeze__11437:0"],
            name="Unsqueeze__11437",
        ),
        helper.make_node(
            "Concat",
            ["Unsqueeze__11434:0", "Unsqueeze__11437:0"],
            ["Concat__11439:0"],
            name="Concat__11439",
            axis=1,
        ),
        helper.make_node("Shape", ["y__10178"], ["Shape_1:0"], name="Shape_1"),
        helper.make_node(
            "Cast",
            ["Shape_1:0"],
            ["Shape_1__10185:0"],
            name="Shape_1__10185",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Slice",
            ["Shape_1__10185:0", "const_axes__11245", "axes_const__9894", "const_axes__11245"],
            ["strided_slice_1:0"],
            name="strided_slice_1",
        ),
        helper.make_node(
            "Squeeze",
            ["strided_slice_1:0", "const_axes__11245"],
            ["strided_slice_1__10189:0"],
            name="strided_slice_1__10189",
        ),
        helper.make_node(
            "Cast",
            ["strided_slice_1__10189:0"],
            ["Cast__11408:0"],
            name="Cast__11408",
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Unsqueeze",
            ["Cast__11408:0", "const_axes__11245"],
            ["Unsqueeze__11410:0"],
            name="Unsqueeze__11410",
        ),
        helper.make_node(
            "Concat",
            ["Unsqueeze__11410:0", "ReduceMax__11420:0"],
            ["Concat__11431:0"],
            name="Concat__11431",
            axis=0,
        ),
        helper.make_node(
            "ConstantOfShape",
            ["Concat__11431:0"],
            ["ConstantOfShape__11432:0"],
            name="ConstantOfShape__11432",
            value=fill_value,
        ),
        helper.make_node(
            "ScatterND",
            ["ConstantOfShape__11432:0", "Concat__11439:0", "UnsortedSegmentMax_TopK__11415:1"],
            ["ScatterND__11442:0"],
            name="ScatterND__11442",
        ),
        helper.make_node(
            "Gather",
            ["Concat__11456:0", "ScatterND__11442:0"],
            ["Gather__11458:0"],
            name="Gather__11458",
        ),
        helper.make_node(
            "ReduceMax",
            ["Gather__11458:0", "axes_const__9894"],
            ["UnsortedSegmentMax_ReduceMax__11463:0"],
            name="UnsortedSegmentMax_ReduceMax__11463",
            keepdims=0,
        ),
        helper.make_node(
            "Gather",
            [
                "UnsortedSegmentMax_ReduceMax__11463:0",
                "DetachedIndices" if detached_final_indices else "Unique__10183_cast:0",
            ],
            ["GatherV2_11:0"],
            name="GatherV2_11",
            axis=0,
        ),
    ]
    graph_inputs = [
        helper.make_tensor_value_info("Reshape:0", TensorProto.INT64, ["N"]),
        helper.make_tensor_value_info("Concat__11456:0", TensorProto.FLOAT, ["M"]),
    ]
    if detached_final_indices:
        graph_inputs.append(
            helper.make_tensor_value_info("DetachedIndices", TensorProto.INT64, ["N"])
        )
    graph = helper.make_graph(
        nodes,
        "segment_max_broadcast_fusion",
        graph_inputs,
        [helper.make_tensor_value_info("GatherV2_11:0", TensorProto.FLOAT, ["N"])],
        initializer=[axes_minus_one, axes_zero, axes_one, range_start, range_delta],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _expected(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.array([values[ids == segment].max() for segment in ids], np.float32)


@pytest.mark.parametrize(
    "ids,values",
    [
        (
            np.array([2, 1, 2, 3, 1], dtype=np.int64),
            np.array([0.4, 2.0, 1.5, -3.0, 0.5], dtype=np.float32),
        ),
        (
            np.array([-7, 100, -7, 100, 42, 42, 42], dtype=np.int64),
            np.array([-5.0, -8.0, -2.0, 9.0, 1.0, 3.0, 2.0], dtype=np.float32),
        ),
        (np.array([9], dtype=np.int64), np.array([1.25], dtype=np.float32)),
    ],
)
def test_segment_max_broadcast_accuracy(ids, values):
    model = _build_segment_max_broadcast_model()
    values_with_sentinel = np.concatenate(
        [values, np.array([np.finfo(np.float32).min], dtype=np.float32)]
    )
    (actual,) = run_model_and_compare(
        model,
        {"Reshape:0": ids, "Concat__11456:0": values_with_sentinel},
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(actual, _expected(ids, values))


def _profile_op_names(model, feeds, tmp_path, prefix, *, disable_cpu_fallback=True):
    so = ort.SessionOptions()
    if disable_cpu_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / prefix)
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
    return {
        event.get("args", {}).get("op_name")
        for event in events
        if event.get("cat") == "Node"
        and event.get("name", "").endswith("_kernel_time")
    }


def test_segment_max_broadcast_fusion_assignment(tmp_path):
    model = _build_segment_max_broadcast_model()
    ids = np.array([3, 1, 3, 2, 1], dtype=np.int64)
    values = np.array([0.5, 4.0, 2.0, -1.0, 3.0], dtype=np.float32)
    feeds = {
        "Reshape:0": ids,
        "Concat__11456:0": np.concatenate(
            [values, np.array([np.finfo(np.float32).min], dtype=np.float32)]
        ),
    }
    op_names = _profile_op_names(model, feeds, tmp_path, "segment_max_broadcast")
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"Unique", "TopK", "ScatterND"} & op_names)


def test_segment_max_broadcast_rejects_broken_final_edge(tmp_path):
    model = _build_segment_max_broadcast_model(detached_final_indices=True)
    ids = np.array([0, 1, 0], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    feeds = {
        "Reshape:0": ids,
        "Concat__11456:0": np.concatenate(
            [values, np.array([np.finfo(np.float32).min], dtype=np.float32)]
        ),
        "DetachedIndices": np.array([0, 1, 0], dtype=np.int64),
    }
    op_names = _profile_op_names(
        model,
        feeds,
        tmp_path,
        "segment_max_rejected",
        disable_cpu_fallback=False,
    )
    assert "Unique" in op_names
    assert not any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
