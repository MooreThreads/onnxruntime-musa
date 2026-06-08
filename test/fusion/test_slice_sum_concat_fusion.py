# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for Slice -> Sum -> Concat fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _int64_initializer(name, values):
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def _build_slice_sum_concat_model() -> bytes:
    initializers = []
    nodes = []

    def add_slice(name, input_name, start, end):
        starts = f"{name}_starts"
        ends = f"{name}_ends"
        axes = f"{name}_axes"
        initializers.extend(
            [
                _int64_initializer(starts, [0, start]),
                _int64_initializer(ends, [9223372036854775807, end]),
                _int64_initializer(axes, [0, 1]),
            ]
        )
        nodes.append(helper.make_node("Slice", [input_name, starts, ends, axes], [name]))
        return name

    a_inputs = [
        add_slice("A0", "X0", 1, 5),
        add_slice("A1", "X1", 3, 7),
        add_slice("A2", "X2", 5, 9),
        add_slice("A3", "X3", 0, 4),
    ]
    b_inputs = [
        add_slice("B0", "X4", 2, 6),
        add_slice("B1", "X5", 4, 8),
        add_slice("B2", "X6", 1, 5),
        add_slice("B3", "X7", 3, 7),
    ]
    nodes.extend(
        [
            helper.make_node("Sum", a_inputs, ["A"]),
            helper.make_node("Sum", b_inputs, ["B"]),
            helper.make_node("Concat", ["D", "A", "B"], ["Y"], axis=1),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "slice_sum_concat_fusion",
        [
            helper.make_tensor_value_info("X0", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("X1", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("X2", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("X3", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("X4", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("X5", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("X6", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("X7", TensorProto.FLOAT, ["batch", 12]),
            helper.make_tensor_value_info("D", TensorProto.FLOAT, ["batch", 2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 10])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_slice_sum_concat_fusion_removes_slice_sum_concat_chain(tmp_path):
    rng = np.random.default_rng(2026)
    feeds = {
        f"X{i}": rng.standard_normal((3, 12)).astype(np.float32)
        for i in range(8)
    }
    feeds["D"] = rng.standard_normal((3, 2)).astype(np.float32)
    model = _build_slice_sum_concat_model()

    (actual,) = run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    expected_a = (
        feeds["X0"][:, 1:5]
        + feeds["X1"][:, 3:7]
        + feeds["X2"][:, 5:9]
        + feeds["X3"][:, 0:4]
    )
    expected_b = (
        feeds["X4"][:, 2:6]
        + feeds["X5"][:, 4:8]
        + feeds["X6"][:, 1:5]
        + feeds["X7"][:, 3:7]
    )
    expected = np.concatenate([feeds["D"], expected_a, expected_b], axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "slice_sum_concat_fusion")
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
    assert not ({"Slice", "Sum", "Concat"} & op_names)
