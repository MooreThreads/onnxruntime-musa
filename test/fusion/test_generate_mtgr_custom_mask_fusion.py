# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""End-to-end tests for the Generate MTGR custom mask fusion."""

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices


DEFAULT_CAPACITY = 37
DEFAULT_PREFIX = 3


def _iota(name: str, shape: tuple[int, ...], capacity: int, corrupt=False):
    values = np.arange(capacity, dtype=np.int64)
    if corrupt:
        values[-1] = values[-1] - 1
    return numpy_helper.from_array(
        values.reshape(shape), name=name
    )


def _build_model(
    prefix: int = DEFAULT_PREFIX,
    capacity: int = DEFAULT_CAPACITY,
    slice_form: str = "double",
    final_cast_type: int = TensorProto.INT32,
    corrupt_capacity_source: bool = False,
) -> bytes:
    initializers = [
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="axis_one"),
        numpy_helper.from_array(np.array(prefix, dtype=np.int64), name="prefix"),
        numpy_helper.from_array(
            np.array([1, 1, 1], dtype=np.int64), name="reshape_111"
        ),
        _iota("row_less_u_iota", (1, capacity, 1), capacity),
        _iota("col_less_u_iota", (1, 1, capacity), capacity),
        _iota(
            "row_ge_u_iota",
            (1, capacity, 1),
            capacity,
            corrupt=corrupt_capacity_source,
        ),
        _iota("row_sub_iota", (1, capacity, 1), capacity),
        _iota("col_sub_iota", (1, 1, capacity), capacity),
        _iota("row_less_l_iota", (1, capacity, 1), capacity),
        _iota("col_ge_u_iota", (1, 1, capacity), capacity),
        _iota("col_less_l_iota", (1, 1, capacity), capacity),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="head_axis"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="scalar_axis"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="slice_start"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), name="slice_axis_2"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), name="slice_axis_3"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_step"),
        numpy_helper.from_array(
            np.array([0, 0], dtype=np.int64), name="slice_starts_2d"
        ),
        numpy_helper.from_array(
            np.array([2, 3], dtype=np.int64), name="slice_axes_2d"
        ),
        numpy_helper.from_array(
            np.array([1, 1], dtype=np.int64), name="slice_steps_2d"
        ),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int32), name="pair"),
    ]

    nodes = []
    source_names = ["view", "cart", "ord", "gp", "target"]
    length_names = []
    for index, source in enumerate(source_names):
        nodes.append(helper.make_node("Shape", [source], [f"shape_{index}"]))
        nodes.append(
            helper.make_node(
                "Gather",
                [f"shape_{index}", "axis_one"],
                [f"length_{index}"],
                axis=0,
            )
        )
        length_names.append(f"length_{index}")

    nodes.extend(
        [
            helper.make_node("Add", [length_names[0], "prefix"], ["u0"]),
            helper.make_node("Add", ["u0", length_names[1]], ["u1"]),
            helper.make_node("Add", ["u1", length_names[2]], ["u2"]),
            helper.make_node("Add", ["u2", length_names[3]], ["U"]),
            helper.make_node("Add", ["U", length_names[4]], ["L"]),
            helper.make_node("Reshape", ["U", "reshape_111"], ["U111"]),
            helper.make_node(
                "Reshape", [length_names[4], "reshape_111"], ["T111"]
            ),
            helper.make_node("Add", ["U111", "T111"], ["L111"]),
            helper.make_node(
                "Less", ["row_less_u_iota", "U111"], ["row_less_u"]
            ),
            helper.make_node(
                "Less", ["col_less_u_iota", "U111"], ["col_less_u"]
            ),
            helper.make_node(
                "And", ["row_less_u", "col_less_u"], ["upstream_square"]
            ),
            helper.make_node(
                "GreaterOrEqual", ["row_ge_u_iota", "U111"], ["row_ge_u"]
            ),
            helper.make_node(
                "And", ["row_ge_u", "col_less_u"], ["target_to_upstream"]
            ),
            helper.make_node("Sub", ["row_sub_iota", "U111"], ["row_offset"]),
            helper.make_node(
                "Cast", ["row_offset"], ["row_offset_i32"], to=TensorProto.INT32
            ),
            helper.make_node("Sub", ["col_sub_iota", "U111"], ["col_offset"]),
            helper.make_node(
                "Cast", ["col_offset"], ["col_offset_i32"], to=TensorProto.INT32
            ),
            helper.make_node(
                "Less", ["row_less_l_iota", "L111"], ["row_less_l"]
            ),
            helper.make_node("And", ["row_ge_u", "row_less_l"], ["row_range"]),
            helper.make_node(
                "GreaterOrEqual", ["col_ge_u_iota", "U111"], ["col_ge_u"]
            ),
            helper.make_node(
                "Less", ["col_less_l_iota", "L111"], ["col_less_l"]
            ),
            helper.make_node("And", ["col_ge_u", "col_less_l"], ["col_range"]),
            helper.make_node("And", ["row_range", "col_range"], ["target_square"]),
            helper.make_node(
                "Equal", ["row_offset_i32", "col_offset_i32"], ["diagonal"]
            ),
            helper.make_node(
                "And", ["target_square", "diagonal"], ["target_diagonal"]
            ),
            helper.make_node(
                "Or", ["upstream_square", "target_to_upstream"], ["upstream_visible"]
            ),
            helper.make_node(
                "Or", ["upstream_visible", "target_diagonal"], ["mask_regions"]
            ),
            helper.make_node("And", ["row_less_l", "col_less_l"], ["valid_square"]),
            helper.make_node("And", ["mask_regions", "valid_square"], ["mask_3d"]),
            helper.make_node("Unsqueeze", ["mask_3d", "head_axis"], ["full_mask"]),
            helper.make_node("Unsqueeze", ["L", "scalar_axis"], ["L1"]),
        ]
    )
    if slice_form == "double":
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [
                        "full_mask",
                        "slice_start",
                        "L1",
                        "slice_axis_2",
                        "slice_step",
                    ],
                    ["mask_rows"],
                ),
                helper.make_node(
                    "Slice",
                    [
                        "mask_rows",
                        "slice_start",
                        "L1",
                        "slice_axis_3",
                        "slice_step",
                    ],
                    ["mask_crop"],
                ),
            ]
        )
    elif slice_form == "single":
        nodes.extend(
            [
                helper.make_node("Concat", ["L1", "L1"], ["L2"], axis=0),
                helper.make_node(
                    "Slice",
                    [
                        "full_mask",
                        "slice_starts_2d",
                        "L2",
                        "slice_axes_2d",
                        "slice_steps_2d",
                    ],
                    ["mask_crop"],
                ),
            ]
        )
    else:
        raise ValueError(f"unsupported slice form: {slice_form}")
    nodes.extend(
        [
            helper.make_node("Cast", ["mask_crop"], ["mask"], to=final_cast_type),
            helper.make_node(
                "Cast", [length_names[4]], ["target_i32"], to=TensorProto.INT32
            ),
            helper.make_node("Mul", ["pair", "target_i32"], ["target_pair"]),
        ]
    )

    inputs = [
        helper.make_tensor_value_info("view", TensorProto.FLOAT, ["B", "V", 256]),
        helper.make_tensor_value_info("cart", TensorProto.FLOAT, ["B", "C", 256]),
        helper.make_tensor_value_info("ord", TensorProto.FLOAT, ["B", "O", 256]),
        helper.make_tensor_value_info("gp", TensorProto.FLOAT, ["B", "G", 256]),
        helper.make_tensor_value_info("target", TensorProto.INT32, ["B", "T"]),
    ]
    outputs = [
        helper.make_tensor_value_info("mask", final_cast_type, None),
        helper.make_tensor_value_info("target_pair", TensorProto.INT32, [2]),
    ]
    graph = helper.make_graph(
        nodes, "generate_mtgr_custom_mask", inputs, outputs, initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _feeds(view=2, cart=3, order=1, gp=4, target=5):
    return {
        "view": np.zeros((1, view, 256), dtype=np.float32),
        "cart": np.zeros((1, cart, 256), dtype=np.float32),
        "ord": np.zeros((1, order, 256), dtype=np.float32),
        "gp": np.zeros((1, gp, 256), dtype=np.float32),
        "target": np.zeros((1, target), dtype=np.int32),
    }


def _profile(model: bytes, feeds, tmp_path):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "generate_mtgr_custom_mask")
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
    op_names = {
        event.get("args", {}).get("op_name")
        for event in events
        if event.get("cat") == "Node"
        and event.get("name", "").endswith("_kernel_time")
    }
    return outputs, op_names


@pytest.mark.parametrize("slice_form", ["double", "single"])
@pytest.mark.parametrize(
    "lengths",
    [
        (2, 3, 1, 4, 5),
        (2, 3, 1, 4, 0),
        (0, 0, 0, 0, 5),
        (34, 0, 0, 0, 0),
        (34, 0, 0, 0, 2),
    ],
)
def test_generate_mtgr_custom_mask_fusion_matches_and_is_exact(
    tmp_path, slice_form, lengths
):
    model = _build_model(slice_form=slice_form)
    feeds = _feeds(*lengths)
    expected = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    actual, op_names = _profile(model, feeds, tmp_path)

    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value, expected_value)
    expected_dimension = min(DEFAULT_PREFIX + sum(lengths), DEFAULT_CAPACITY)
    assert actual[0].shape == (1, 1, expected_dimension, expected_dimension)
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not (
        {
            "Shape",
            "Gather",
            "Add",
            "Reshape",
            "Less",
            "GreaterOrEqual",
            "And",
            "Or",
            "Sub",
            "Equal",
        }
        & op_names
    )
    assert "Slice" not in op_names
    assert "Cast" in op_names


def test_generate_mtgr_custom_mask_uses_graph_capacity_and_prefix(tmp_path):
    capacity = 29
    prefix = 5
    model = _build_model(capacity=capacity, prefix=prefix)
    feeds = _feeds()
    expected = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    actual, op_types = _profile(model, feeds, tmp_path)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value, expected_value)
    dimension = min(prefix + 2 + 3 + 1 + 4 + 5, capacity)
    assert actual[0].shape == (1, 1, dimension, dimension)
    assert "Slice" not in op_types


def test_generate_mtgr_custom_mask_rejects_invalid_capacity_source(tmp_path):
    model = _build_model(corrupt_capacity_source=True)
    optimized_path = tmp_path / "generate_mtgr_invalid_iota.optimized.onnx"
    so = ort.SessionOptions()
    so.optimized_model_filepath = str(optimized_path)
    so.add_provider_for_devices(musa_devices(), {})
    ort.InferenceSession(model, sess_options=so)
    optimized = onnx.load(optimized_path, load_external_data=False)
    op_types = {node.op_type for node in optimized.graph.node}
    assert "Less" in op_types
    assert not any(str(op).startswith("MUSAExecutionProvider_") for op in op_types)


def test_generate_mtgr_custom_mask_rejects_wrong_final_cast(tmp_path):
    model = _build_model(final_cast_type=TensorProto.INT64)
    feeds = _feeds()
    expected = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    actual, op_types = _profile(model, feeds, tmp_path)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value, expected_value)
    # The strict phase-2 tail is rejected. The compatible phase-1 core may
    # still fuse, but Slice and the non-int32 final Cast must remain outside.
    assert "Slice" in op_types
    assert "Cast" in op_types
