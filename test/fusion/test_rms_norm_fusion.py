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
"""End-to-end tests for RMSNorm fusion lowerings."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    float32_to_bfloat16_bits,
    musa_devices,
    run_model_and_compare,
    run_with_iobinding,
)


def _bf16_initializer(name: str, values: np.ndarray):
    return helper.make_tensor(
        name,
        TensorProto.BFLOAT16,
        values.shape,
        values.astype(np.uint16).tobytes(),
        raw=True,
    )


def _build_rms_norm_model(epsilon: float = 1.0e-6) -> bytes:
    axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes")
    eps = numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name="eps")
    gamma = numpy_helper.from_array(
        np.linspace(0.5, 1.5, num=5, dtype=np.float32), name="gamma"
    )
    nodes = [
        helper.make_node("Mul", ["X", "X"], ["Square"], name="rms_norm/Square"),
        helper.make_node(
            "ReduceMean", ["Square", "axes"], ["Mean"], keepdims=1, name="rms_norm/Mean"
        ),
        helper.make_node("Add", ["Mean", "eps"], ["MeanEps"], name="rms_norm/add"),
        helper.make_node("Sqrt", ["MeanEps"], ["Denom"], name="rms_norm/Sqrt"),
        helper.make_node("Div", ["X", "Denom"], ["Normed"], name="rms_norm/truediv"),
        helper.make_node("Mul", ["Normed", "gamma"], ["Y"], name="rms_norm/mul"),
    ]
    graph = helper.make_graph(
        nodes,
        "rms_norm_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[axes, eps, gamma],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_typed_rms_norm_model(tensor_type: int, gamma, epsilon) -> bytes:
    axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes")
    if tensor_type == TensorProto.BFLOAT16:
        gamma_init = _bf16_initializer("gamma", gamma)
        eps_init = _bf16_initializer("eps", np.asarray(epsilon))
    else:
        gamma_init = numpy_helper.from_array(gamma, name="gamma")
        eps_init = numpy_helper.from_array(np.asarray(epsilon), name="eps")
    nodes = [
        helper.make_node("Mul", ["X", "X"], ["Square"], name="rms_norm/Square"),
        helper.make_node(
            "ReduceMean", ["Square", "axes"], ["Mean"], keepdims=1, name="rms_norm/Mean"
        ),
        helper.make_node("Add", ["Mean", "eps"], ["MeanEps"], name="rms_norm/add"),
        helper.make_node("Sqrt", ["MeanEps"], ["Denom"], name="rms_norm/Sqrt"),
        helper.make_node("Div", ["X", "Denom"], ["Normed"], name="rms_norm/truediv"),
        helper.make_node("Mul", ["Normed", "gamma"], ["Y"], name="rms_norm/mul"),
    ]
    graph = helper.make_graph(
        nodes,
        "typed_rms_norm_fusion",
        [helper.make_tensor_value_info("X", tensor_type, [2, 3, 5])],
        [helper.make_tensor_value_info("Y", tensor_type, [2, 3, 5])],
        initializer=[axes, eps_init, gamma_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_pow_rms_norm_model() -> bytes:
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two")
    eps = numpy_helper.from_array(np.array(1.0e-6, dtype=np.float32), name="eps")
    gamma = numpy_helper.from_array(
        np.linspace(0.5, 1.5, num=5, dtype=np.float32), name="gamma"
    )
    nodes = [
        helper.make_node("Pow", ["X", "two"], ["Square"], name="rms_norm/Pow"),
        helper.make_node(
            "ReduceMean", ["Square"], ["Mean"], axes=[-1], keepdims=1,
            name="rms_norm/MeanAttr"
        ),
        helper.make_node("Add", ["Mean", "eps"], ["MeanEps"], name="rms_norm/add"),
        helper.make_node("Sqrt", ["MeanEps"], ["Denom"], name="rms_norm/Sqrt"),
        helper.make_node("Div", ["X", "Denom"], ["Normed"], name="rms_norm/truediv"),
        helper.make_node("Mul", ["Normed", "gamma"], ["Y"], name="rms_norm/mul"),
    ]
    graph = helper.make_graph(
        nodes,
        "rms_norm_pow_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[two, eps, gamma],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _profile_op_names(
    model: bytes,
    feeds,
    profile_prefix: str,
    feed_types=None,
    outputs=None,
):
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = profile_prefix
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    if outputs is None:
        session.run(None, feeds)
    else:
        output_dtypes = {
            TensorProto.BFLOAT16: np.uint16,
            TensorProto.FLOAT16: np.float16,
            TensorProto.DOUBLE: np.float64,
            TensorProto.FLOAT: np.float32,
        }
        io_binding = session.io_binding()
        feed_types = feed_types or {}
        for name, arr in feeds.items():
            elem_type = feed_types.get(name, helper.np_dtype_to_tensor_dtype(arr.dtype))
            io_binding.bind_input(name, "cpu", 0, elem_type, arr.shape, arr.ctypes.data)
        output_buffers = []
        for name, elem_type, shape in outputs:
            output = np.empty(tuple(shape), dtype=output_dtypes[elem_type])
            io_binding.bind_output(name, "cpu", 0, elem_type, output.shape, output.ctypes.data)
            output_buffers.append(output)
        session.run_with_iobinding(io_binding)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    return {
        e.get("args", {}).get("op_name")
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    }


def _assert_rms_norm_fused(op_names):
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"ReduceMean", "Add", "Sqrt", "Div", "Mul"} & op_names)


def _rms_norm_reference(x: np.ndarray, gamma: np.ndarray, epsilon: float) -> np.ndarray:
    x_f32 = x.astype(np.float32)
    gamma_f32 = gamma.astype(np.float32)
    return (
        x_f32
        * np.reciprocal(
            np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + epsilon)
        )
        * gamma_f32
    )


def test_rms_norm_fusion(tmp_path):
    model = _build_rms_norm_model()
    x = np.linspace(-2.0, 2.0, num=2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    feeds = {"X": x}
    (actual,) = run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    gamma = np.linspace(0.5, 1.5, num=5, dtype=np.float32)
    expected = x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6))
    expected = expected * gamma
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    op_names = _profile_op_names(model, feeds, str(tmp_path / "rms_norm_fusion"))
    _assert_rms_norm_fused(op_names)


def test_pow_rms_norm_fusion(tmp_path):
    model = _build_pow_rms_norm_model()
    x = np.linspace(-2.0, 2.0, num=2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    feeds = {"X": x}
    (actual,) = run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    gamma = np.linspace(0.5, 1.5, num=5, dtype=np.float32)
    expected = x * np.reciprocal(
        np.sqrt(np.mean(np.power(x, 2), axis=-1, keepdims=True) + 1e-6)
    ) * gamma
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    op_names = _profile_op_names(model, feeds, str(tmp_path / "rms_norm_pow_fusion"))
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul"} & op_names)


def test_rms_norm_fusion_caches_initializer_epsilon(tmp_path):
    epsilon = 0.25
    model = _build_rms_norm_model(epsilon)
    x = np.linspace(-1.0, 1.0, num=2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    feeds = {"X": x}
    (actual,) = run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    gamma = np.linspace(0.5, 1.5, num=5, dtype=np.float32)
    expected = x * np.reciprocal(
        np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)
    ) * gamma
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    op_names = _profile_op_names(
        model, feeds, str(tmp_path / "rms_norm_cached_epsilon")
    )
    _assert_rms_norm_fused(op_names)


def test_rms_norm_fusion_float16(tmp_path):
    epsilon = np.array(1.0e-4, dtype=np.float16)
    gamma = np.linspace(0.5, 1.5, num=5, dtype=np.float16)
    model = _build_typed_rms_norm_model(TensorProto.FLOAT16, gamma, epsilon)
    x = np.linspace(-2.0, 2.0, num=2 * 3 * 5, dtype=np.float16).reshape(2, 3, 5)
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.FLOAT16},
        [("Y", TensorProto.FLOAT16, x.shape)],
        use_musa=True,
    )
    expected = _rms_norm_reference(x, gamma, float(epsilon)).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
    op_names = _profile_op_names(
        model,
        {"X": x},
        str(tmp_path / "rms_norm_float16"),
        {"X": TensorProto.FLOAT16},
        [("Y", TensorProto.FLOAT16, x.shape)],
    )
    _assert_rms_norm_fused(op_names)


def test_rms_norm_fusion_bfloat16(tmp_path):
    epsilon_f32 = np.array(1.0e-4, dtype=np.float32)
    gamma_f32 = np.linspace(0.5, 1.5, num=5, dtype=np.float32)
    x_f32 = np.linspace(-2.0, 2.0, num=2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    gamma = float32_to_bfloat16_bits(gamma_f32)
    epsilon = float32_to_bfloat16_bits(epsilon_f32)
    x = float32_to_bfloat16_bits(x_f32)
    model = _build_typed_rms_norm_model(TensorProto.BFLOAT16, gamma, epsilon)
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
        use_musa=True,
    )
    expected = _rms_norm_reference(
        bfloat16_bits_to_float32(x),
        bfloat16_bits_to_float32(gamma),
        float(bfloat16_bits_to_float32(epsilon)),
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), expected, rtol=6e-2, atol=6e-2
    )
    op_names = _profile_op_names(
        model,
        {"X": x},
        str(tmp_path / "rms_norm_bfloat16"),
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, x.shape)],
    )
    _assert_rms_norm_fused(op_names)


def test_rms_norm_fusion_float64(tmp_path):
    epsilon = np.array(1.0e-6, dtype=np.float64)
    gamma = np.linspace(0.5, 1.5, num=5, dtype=np.float64)
    model = _build_typed_rms_norm_model(TensorProto.DOUBLE, gamma, epsilon)
    x = np.linspace(-2.0, 2.0, num=2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5)
    (actual,) = run_model_and_compare(model, {"X": x}, rtol=1e-5, atol=1e-5)
    expected = _rms_norm_reference(x, gamma, float(epsilon))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    op_names = _profile_op_names(model, {"X": x}, str(tmp_path / "rms_norm_float64"))
    _assert_rms_norm_fused(op_names)
