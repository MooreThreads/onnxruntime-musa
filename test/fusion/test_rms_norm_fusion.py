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
"""End-to-end test for the Mul/ReduceMean/Add/Sqrt/Div/Mul RMSNorm fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_rms_norm_model() -> bytes:
    axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes")
    eps = numpy_helper.from_array(np.array(1.0e-6, dtype=np.float32), name="eps")
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


def test_rms_norm_fusion(tmp_path):
    model = _build_rms_norm_model()
    x = np.linspace(-2.0, 2.0, num=2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    feeds = {"X": x}

    (actual,) = run_model_and_compare(model, feeds, rtol=1e-5, atol=1e-5)
    gamma = np.linspace(0.5, 1.5, num=5, dtype=np.float32)
    expected = x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6))
    expected = expected * gamma
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "rms_norm_fusion")
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
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"ReduceMean", "Add", "Sqrt", "Div", "Mul"} & op_names)
