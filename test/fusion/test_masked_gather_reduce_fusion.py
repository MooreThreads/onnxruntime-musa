# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for NonZero indices feeding Gather + Reduce."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_masked_gather_reduce_model(reduce_op: str) -> bytes:
    squeeze_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="squeeze_axes"
    )
    reduce_axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="reduce_axes"
    )
    nodes = [
        helper.make_node("NonZero", ["M"], ["nz"]),
        helper.make_node("Transpose", ["nz"], ["nz_t"]),
        helper.make_node("Squeeze", ["nz_t", "squeeze_axes"], ["idx"]),
        helper.make_node("Gather", ["X", "idx"], ["selected"], axis=0),
        helper.make_node(
            reduce_op,
            ["selected", "reduce_axes"],
            ["Y"],
            keepdims=0,
            noop_with_empty_axes=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "masked_gather_reduce_fusion_profile",
        [
            helper.make_tensor_value_info("M", TensorProto.BOOL, ["N"]),
            helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [])],
        initializer=[squeeze_axes, reduce_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


@pytest.mark.parametrize("reduce_op", ["ReduceProd", "ReduceMean"])
def test_masked_gather_reduce_fusion_removes_nonzero_chain(tmp_path, reduce_op):
    model = _build_masked_gather_reduce_model(reduce_op)
    feeds = {
        "M": np.array([True, False, True, True, False], dtype=np.bool_),
        "X": np.array([1.5, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    }

    (actual,) = run_model_and_compare(model, feeds)
    selected = feeds["X"][feeds["M"]]
    expected = (
        np.prod(selected).astype(np.float32)
        if reduce_op == "ReduceProd"
        else np.mean(selected).astype(np.float32)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "masked_gather_reduce_fusion")
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
    assert not (
        {"NonZero", "Transpose", "Squeeze", "Gather", reduce_op} & op_names
    )
