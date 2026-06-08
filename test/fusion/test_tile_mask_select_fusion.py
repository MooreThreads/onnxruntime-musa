# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for Tile(mask) feeding mask/inverse select arithmetic."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_tile_mask_select_model() -> bytes:
    repeats = numpy_helper.from_array(np.array([1, 4], dtype=np.int64), name="repeats")
    nodes = [
        helper.make_node("Tile", ["M", "repeats"], ["Mt"]),
        helper.make_node("Cast", ["Mt"], ["Mf"], to=TensorProto.FLOAT),
        helper.make_node("Not", ["Mt"], ["Mn"]),
        helper.make_node("Cast", ["Mn"], ["Mnf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["Mf", "A"], ["MA"]),
        helper.make_node("Mul", ["B", "Mnf"], ["MB"]),
        helper.make_node("Add", ["MA", "MB"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tile_mask_select_fusion_profile",
        [
            helper.make_tensor_value_info("M", TensorProto.BOOL, ["N", 1]),
            helper.make_tensor_value_info("A", TensorProto.FLOAT, ["N", 4]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, ["N", 4]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[repeats],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_tile_mask_select_fusion_removes_tile_arithmetic_chain(tmp_path):
    model = _build_tile_mask_select_model()
    feeds = {
        "M": np.array([[True], [False], [True]], dtype=np.bool_),
        "A": np.arange(12, dtype=np.float32).reshape(3, 4),
        "B": (np.arange(12, dtype=np.float32).reshape(3, 4) + 100.0),
    }

    (actual,) = run_model_and_compare(model, feeds)
    expected = np.where(feeds["M"], feeds["A"], feeds["B"])
    np.testing.assert_array_equal(actual, expected)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "tile_mask_select_fusion")
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

    node_events = [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    ]
    op_names = {e.get("args", {}).get("op_name") for e in node_events}
    providers = {e.get("args", {}).get("provider") for e in node_events}
    assert "MUSAExecutionProvider" in providers
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"Tile", "Cast", "Not", "Mul", "Add"} & op_names)
