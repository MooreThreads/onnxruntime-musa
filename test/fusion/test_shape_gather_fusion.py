# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the Plugin EP Shape -> Cast -> Gather fusion pattern."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_shape_cast_gather_model() -> bytes:
    index = numpy_helper.from_array(np.array([1], dtype=np.int64), name="index")
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["shape_i32"], to=TensorProto.INT32),
        helper.make_node("Gather", ["shape_i32", "index"], ["Y"], axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_cast_gather_fusion_profile",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", "M", "K"])],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [None])],
        initializer=[index],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_shape_cast_gather_fusion_with_initializer_indices(tmp_path):
    model = _build_shape_cast_gather_model()
    x = np.zeros((2, 3, 4), dtype=np.float32)

    (actual,) = run_model_and_compare(model, {"X": x})
    np.testing.assert_array_equal(actual, np.array([3], dtype=np.int32))

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "shape_cast_gather_fusion")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    session.run(None, {"X": x})
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
    assert not ({"Shape", "Cast", "Gather"} & op_names)
