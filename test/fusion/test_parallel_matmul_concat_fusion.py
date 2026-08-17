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
"""End-to-end tests for parallel MatMul -> Unsqueeze -> Concat fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    float32_to_bfloat16_bits,
    musa_devices,
    run_model_and_compare,
    run_with_iobinding,
)


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


def _profile_musa_node_names_iobinding(
    model_bytes, feeds, feed_types, outputs, branch_count
):
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for profiling")

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = (
        f"parallel_matmul_concat_fusion_iobinding_{branch_count}"
    )
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(model_bytes, sess_options=session_options)
    io_binding = session.io_binding()
    for name, value in feeds.items():
        io_binding.bind_input(
            name, "cpu", 0, feed_types[name], value.shape, value.ctypes.data
        )
    for name, elem_type, shape in outputs:
        output = np.empty(tuple(shape), dtype=np.float32)
        io_binding.bind_output(
            name, "cpu", 0, elem_type, output.shape, output.ctypes.data
        )
    session.run_with_iobinding(io_binding)

    profile_path = Path(session.end_profiling())
    try:
        events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    return [event.get("name", "") for event in events if event.get("cat") == "Node"]


def _build_model(branch_count, initializer_weights=False):
    rng = np.random.default_rng(branch_count)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    weights = {
        f"W{i}": rng.standard_normal((8, 5)).astype(np.float32)
        for i in range(branch_count)
    }
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    initializers = [axes]

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

    feeds = {"X": x}
    if not initializer_weights:
        feeds.update(weights)

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 8])]
    if initializer_weights:
        initializers.extend(
            numpy_helper.from_array(weight, name=name)
            for name, weight in weights.items()
        )
    else:
        inputs.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(weight.shape))
            for name, weight in weights.items()
        )

    graph = helper.make_graph(
        nodes,
        f"parallel_matmul_concat_fusion_{branch_count}_graph",
        inputs,
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["batch", branch_count, 5])],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


def _build_float16_model(branch_count):
    rng = np.random.default_rng(branch_count + 100)
    x = rng.standard_normal((3, 8)).astype(np.float16)
    weights = {
        f"W{i}": rng.standard_normal((8, 5)).astype(np.float16)
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
            helper.make_tensor_value_info(matmul_output, TensorProto.FLOAT16, ["batch", 5])
        )
        value_info.append(
            helper.make_tensor_value_info(
                unsqueeze_output, TensorProto.FLOAT16, ["batch", 1, 5]
            )
        )
    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=1))
    nodes.append(helper.make_node("Cast", ["Y"], ["Z"], to=TensorProto.FLOAT))
    value_info.append(
        helper.make_tensor_value_info(
            "Y", TensorProto.FLOAT16, ["batch", branch_count, 5]
        )
    )

    feeds = {"X": x, **weights}
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["batch", 8]),
        *[
            helper.make_tensor_value_info(name, TensorProto.FLOAT16, list(weight.shape))
            for name, weight in weights.items()
        ],
    ]
    graph = helper.make_graph(
        nodes,
        f"parallel_matmul_concat_float16_fusion_{branch_count}_graph",
        inputs,
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["batch", branch_count, 5])],
        initializer=[axes],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString(), feeds


def _build_bfloat16_model(branch_count):
    rng = np.random.default_rng(branch_count + 200)
    x_f32 = rng.standard_normal((3, 8)).astype(np.float32)
    weight_f32 = {
        f"W{i}": rng.standard_normal((8, 5)).astype(np.float32)
        for i in range(branch_count)
    }
    feeds = {"X": float32_to_bfloat16_bits(x_f32)}
    feeds.update(
        {name: float32_to_bfloat16_bits(value) for name, value in weight_f32.items()}
    )
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
            helper.make_tensor_value_info(matmul_output, TensorProto.BFLOAT16, ["batch", 5])
        )
        value_info.append(
            helper.make_tensor_value_info(
                unsqueeze_output, TensorProto.BFLOAT16, ["batch", 1, 5]
            )
        )
    nodes.append(helper.make_node("Concat", concat_inputs, ["Y"], axis=1))
    nodes.append(helper.make_node("Cast", ["Y"], ["Z"], to=TensorProto.FLOAT))
    value_info.append(
        helper.make_tensor_value_info(
            "Y", TensorProto.BFLOAT16, ["batch", branch_count, 5]
        )
    )

    inputs = [
        helper.make_tensor_value_info("X", TensorProto.BFLOAT16, ["batch", 8]),
        *[
            helper.make_tensor_value_info(name, TensorProto.BFLOAT16, list(value.shape))
            for name, value in weight_f32.items()
        ],
    ]
    graph = helper.make_graph(
        nodes,
        f"parallel_matmul_concat_bfloat16_fusion_{branch_count}_graph",
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


def test_parallel_matmul_concat_fusion_initializer_weights():
    model, feeds = _build_model(4, initializer_weights=True)
    run_model_and_compare(model, feeds, rtol=1e-3, atol=1e-3)
    node_names = _profile_musa_node_names(model, feeds, 4)
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(
        name.startswith(("MatMul_", "Unsqueeze_", "Concat_")) for name in node_names
    )


def test_parallel_matmul_concat_fusion_float16():
    model, feeds = _build_float16_model(4)
    expected = np.stack(
        [feeds["X"].astype(np.float32) @ feeds[f"W{i}"].astype(np.float32) for i in range(4)],
        axis=1,
    )
    outputs = run_model_and_compare(model, feeds, rtol=4e-2, atol=4e-2)
    np.testing.assert_allclose(outputs[0], expected, rtol=4e-2, atol=4e-2)
    node_names = _profile_musa_node_names(model, feeds, 4)
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(
        name.startswith(("MatMul_", "Unsqueeze_", "Concat_")) for name in node_names
    )


def test_parallel_matmul_concat_fusion_bfloat16():
    model, feeds = _build_bfloat16_model(4)
    outputs = run_with_iobinding(
        model,
        feeds,
        {name: TensorProto.BFLOAT16 for name in feeds},
        [("Z", TensorProto.FLOAT, (3, 4, 5))],
        use_musa=True,
    )
    x = bfloat16_bits_to_float32(feeds["X"])
    expected = np.stack(
        [x @ bfloat16_bits_to_float32(feeds[f"W{i}"]) for i in range(4)], axis=1
    )
    np.testing.assert_allclose(outputs[0], expected, rtol=7e-2, atol=7e-2)

    node_names = _profile_musa_node_names_iobinding(
        model,
        feeds,
        {name: TensorProto.BFLOAT16 for name in feeds},
        [("Z", TensorProto.FLOAT, (3, 4, 5))],
        4,
    )
    assert any(name.startswith("MUSAExecutionProvider_") for name in node_names)
    assert not any(
        name.startswith(("MatMul_", "Unsqueeze_", "Concat_")) for name in node_names
    )
