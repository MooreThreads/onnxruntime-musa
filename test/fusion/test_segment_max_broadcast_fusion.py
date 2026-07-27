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
"""End-to-end tests for the SegmentMaxBroadcast fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _rename_graph_values(graph, prefix):
    names = {
        name for node in graph.node for name in (*node.input, *node.output) if name
    }
    names.update(value.name for value in graph.input)
    names.update(value.name for value in graph.output)
    names.update(value.name for value in graph.value_info)
    names.update(initializer.name for initializer in graph.initializer)
    mapping = {
        name: f"{prefix}value_{index}" for index, name in enumerate(sorted(names))
    }

    for value in (*graph.input, *graph.output, *graph.value_info):
        value.name = mapping[value.name]
    for initializer in graph.initializer:
        initializer.name = mapping[initializer.name]
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = mapping[name]
        for index, name in enumerate(node.output):
            if name:
                node.output[index] = mapping[name]
    return mapping


def _build_segment_max_broadcast_model(
    *,
    detached_final_indices=False,
    downstream_consumer=False,
    rename_values=False,
    segment_id_type=TensorProto.INT64,
):
    axis_minus_one = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name="axis_minus_one"
    )
    axis_zero = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axis_zero")
    axis_one = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis_one")
    range_start = numpy_helper.from_array(
        np.array(0, dtype=np.int64), name="range_start"
    )
    range_step = numpy_helper.from_array(np.array(1, dtype=np.int64), name="range_step")
    fill_value = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name="fill_value"
    )
    output_floor = numpy_helper.from_array(
        np.array(-1.0e30, dtype=np.float32), name="output_floor"
    )

    nodes = [
        helper.make_node(
            "Unique",
            ["segment_ids"],
            [
                "unique_values",
                "unique_indices",
                "unique_inverse",
                "unique_counts",
            ],
            sorted=0,
        ),
        helper.make_node(
            "Cast",
            ["unique_inverse"],
            ["inverse_indices_i32"],
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Cast",
            ["inverse_indices_i32"],
            ["inverse_indices_i64"],
            to=TensorProto.INT64,
        ),
        helper.make_node("Shape", ["inverse_indices_i64"], ["inverse_indices_shape"]),
        helper.make_node(
            "TopK",
            ["inverse_indices_i64", "inverse_indices_shape"],
            ["sorted_segment_ids", "sorted_positions"],
            axis=0,
            largest=0,
            sorted=1,
        ),
        helper.make_node(
            "Unsqueeze",
            ["sorted_segment_ids", "axis_minus_one"],
            ["sorted_segment_ids_column"],
        ),
        helper.make_node(
            "Unique",
            ["sorted_segment_ids"],
            ["ordered_ids", "", "ordered_inverse", "ordered_counts"],
            axis=0,
            sorted=1,
        ),
        helper.make_node(
            "ReduceMax",
            ["ordered_counts", "axis_zero"],
            ["largest_group_size"],
            keepdims=1,
        ),
        helper.make_node(
            "Gather",
            ["ordered_counts", "ordered_inverse"],
            ["expanded_counts"],
        ),
        helper.make_node(
            "Squeeze",
            ["inverse_indices_shape", "axis_zero"],
            ["element_count"],
        ),
        helper.make_node(
            "Range",
            ["range_start", "element_count", "range_step"],
            ["linear_positions"],
        ),
        helper.make_node(
            "Mod",
            ["linear_positions", "expanded_counts"],
            ["group_positions"],
        ),
        helper.make_node(
            "Unsqueeze",
            ["group_positions", "axis_minus_one"],
            ["group_positions_column"],
        ),
        helper.make_node(
            "Concat",
            ["sorted_segment_ids_column", "group_positions_column"],
            ["scatter_indices"],
            axis=1,
        ),
        helper.make_node("Shape", ["unique_values"], ["unique_values_shape"]),
        helper.make_node(
            "Cast",
            ["unique_values_shape"],
            ["unique_values_shape_i32"],
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Slice",
            ["unique_values_shape_i32", "axis_zero", "axis_one", "axis_zero"],
            ["unique_count_vector_i32"],
        ),
        helper.make_node(
            "Squeeze",
            ["unique_count_vector_i32", "axis_zero"],
            ["unique_count_i32"],
        ),
        helper.make_node(
            "Cast",
            ["unique_count_i32"],
            ["unique_count_i64"],
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Unsqueeze",
            ["unique_count_i64", "axis_zero"],
            ["unique_count_vector_i64"],
        ),
        helper.make_node(
            "Concat",
            ["unique_count_vector_i64", "largest_group_size"],
            ["scatter_shape"],
            axis=0,
        ),
        helper.make_node(
            "ConstantOfShape",
            ["scatter_shape"],
            ["scatter_base"],
            value=fill_value,
        ),
        helper.make_node(
            "ScatterND",
            ["scatter_base", "scatter_indices", "sorted_positions"],
            ["position_map"],
        ),
        helper.make_node(
            "Gather",
            ["values_with_sentinel", "position_map"],
            ["grouped_values"],
        ),
        helper.make_node(
            "ReduceMax",
            ["grouped_values", "axis_one"],
            ["per_group_max"],
            keepdims=0,
        ),
        helper.make_node(
            "Gather",
            [
                "per_group_max",
                "detached_indices" if detached_final_indices else "inverse_indices_i32",
            ],
            ["segment_max_output"],
            axis=0,
        ),
    ]
    graph_output_name = "segment_max_output"
    if downstream_consumer:
        graph_output_name = "model_output"
        nodes.append(
            helper.make_node(
                "Max",
                ["segment_max_output", "output_floor"],
                [graph_output_name],
            )
        )
    for index, node in enumerate(nodes):
        node.name = f"node_{index}"

    graph_inputs = [
        helper.make_tensor_value_info("segment_ids", segment_id_type, ["N"]),
        helper.make_tensor_value_info("values_with_sentinel", TensorProto.FLOAT, ["M"]),
    ]
    if detached_final_indices:
        graph_inputs.append(
            helper.make_tensor_value_info("detached_indices", TensorProto.INT64, ["N"])
        )
    graph = helper.make_graph(
        nodes,
        "segment_max_broadcast_fusion",
        graph_inputs,
        [helper.make_tensor_value_info(graph_output_name, TensorProto.FLOAT, ["N"])],
        initializer=[
            axis_minus_one,
            axis_zero,
            axis_one,
            range_start,
            range_step,
            output_floor,
        ],
    )
    feed_names = {
        "segment_ids": "segment_ids",
        "values": "values_with_sentinel",
    }
    if detached_final_indices:
        feed_names["detached_indices"] = "detached_indices"
    if rename_values:
        mapping = _rename_graph_values(graph, "renamed_")
        feed_names = {role: mapping[name] for role, name in feed_names.items()}

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feed_names


def _expected(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.array([values[ids == segment].max() for segment in ids], np.float32)


@pytest.mark.parametrize("rename_values", [False, True])
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
def test_segment_max_broadcast_accuracy(ids, values, rename_values):
    model, names = _build_segment_max_broadcast_model(rename_values=rename_values)
    values_with_sentinel = np.concatenate(
        [values, np.array([np.finfo(np.float32).min], dtype=np.float32)]
    )
    (actual,) = run_model_and_compare(
        model,
        {names["segment_ids"]: ids, names["values"]: values_with_sentinel},
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
        if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")
    }


@pytest.mark.parametrize("rename_values", [False, True])
@pytest.mark.parametrize("downstream_consumer", [False, True])
def test_segment_max_broadcast_fusion_assignment(
    tmp_path, downstream_consumer, rename_values
):
    model, names = _build_segment_max_broadcast_model(
        downstream_consumer=downstream_consumer,
        rename_values=rename_values,
    )
    ids = np.array([3, 1, 3, 2, 1], dtype=np.int64)
    values = np.array([0.5, 4.0, 2.0, -1.0, 3.0], dtype=np.float32)
    feeds = {
        names["segment_ids"]: ids,
        names["values"]: np.concatenate(
            [values, np.array([np.finfo(np.float32).min], dtype=np.float32)]
        ),
    }
    op_names = _profile_op_names(
        model,
        feeds,
        tmp_path,
        f"segment_max_broadcast_{downstream_consumer}_{rename_values}",
    )
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"Unique", "TopK", "ScatterND"} & op_names)


@pytest.mark.parametrize("rename_values", [False, True])
def test_segment_max_broadcast_rejects_broken_final_edge(tmp_path, rename_values):
    model, names = _build_segment_max_broadcast_model(
        detached_final_indices=True,
        rename_values=rename_values,
    )
    ids = np.array([0, 1, 0], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    feeds = {
        names["segment_ids"]: ids,
        names["values"]: np.concatenate(
            [values, np.array([np.finfo(np.float32).min], dtype=np.float32)]
        ),
        names["detached_indices"]: np.array([0, 1, 0], dtype=np.int64),
    }
    op_names = _profile_op_names(
        model,
        feeds,
        tmp_path,
        f"segment_max_rejected_{rename_values}",
        disable_cpu_fallback=False,
    )
    assert "Unique" in op_names
    assert not any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)


def test_segment_max_broadcast_rejects_unsupported_id_type(tmp_path):
    model, names = _build_segment_max_broadcast_model(segment_id_type=TensorProto.INT32)
    ids = np.array([0, 1, 0], dtype=np.int32)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    feeds = {
        names["segment_ids"]: ids,
        names["values"]: np.concatenate(
            [values, np.array([np.finfo(np.float32).min], dtype=np.float32)]
        ),
    }
    op_names = _profile_op_names(
        model,
        feeds,
        tmp_path,
        "segment_max_unsupported_id_type",
        disable_cpu_fallback=False,
    )
    assert "Unique" in op_names
    assert not any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
