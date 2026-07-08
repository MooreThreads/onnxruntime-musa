# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for parallel MatMul -> Unsqueeze -> Concat fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _profile_musa_node_names(model_bytes, feeds, branch_count):
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for profiling")

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = (
        f"parallel_matmul_concat_fusion_{branch_count}"
    )
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model_bytes, sess_options=session_options)
    session.run(None, dict(feeds))

    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


def _build_model(branch_count):
    rng = np.random.default_rng(branch_count)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    weights = {
        f"W{i}": rng.standard_normal((8, 5)).astype(np.float32)
        for i in range(branch_count)
    }
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")

    nodes = []
    concat_inputs = []
    value_info = []
    for i in range(branch_count):
        matmul_output = f"M{i}"
        unsqueeze_output = f"U{i}"
        nodes.append(helper.make_node("MatMul", ["X", f"W{i}"], [matmul_output]))
        nodes.append(
            helper.make_node("Unsqueeze", [matmul_output, "axes"], [unsqueeze_output])
        )
        concat_inputs.append(unsqueeze_output)
        value_info.append(
            helper.make_tensor_value_info(matmul_output, TensorProto.FLOAT, ["batch", 5])
        )
        value_info.append(
            helper.make_tensor_value_info(
                unsqueeze_output, TensorProto.FLOAT, ["batch", 1, 5]
            )
        )
    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=1))
    nodes.append(helper.make_node("Softmax", ["Y"], ["Z"], axis=1))
    value_info.append(
        helper.make_tensor_value_info(
            "Y", TensorProto.FLOAT, ["batch", branch_count, 5]
        )
    )

    feeds = {"X": x, **weights}
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 8])]
    inputs.extend(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(weight.shape))
        for name, weight in weights.items()
    )
    graph = helper.make_graph(
        nodes,
        f"parallel_matmul_concat_fusion_{branch_count}_graph",
        inputs,
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["batch", branch_count, 5])],
        initializer=[axes],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


@pytest.mark.parametrize("branch_count", [3, 4, 6])
def test_parallel_matmul_concat_fusion(branch_count):
    model, feeds = _build_model(branch_count)
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_musa_node_names(model, feeds, branch_count)
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(
        name.startswith(("MatMul_", "Unsqueeze_", "Concat_")) for name in node_names
    )
