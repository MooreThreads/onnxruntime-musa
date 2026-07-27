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
"""End-to-end test for the Max/Add/Log/Mul MathConcatLog fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_math_concat_log_model() -> bytes:
    max_value = numpy_helper.from_array(
        np.array(1.0e-6, dtype=np.float32), name="max_value"
    )
    add_value = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="add")
    scale_value = numpy_helper.from_array(
        np.array(0.5, dtype=np.float32), name="scale"
    )
    nodes = [
        helper.make_node(
            "Max", ["X", "max_value"], ["Clipped"], name="MathConcatLog/Maximum"
        ),
        helper.make_node("Add", ["Clipped", "add"], ["Shifted"], name="MathConcatLog/add"),
        helper.make_node("Log", ["Shifted"], ["Logged"], name="MathConcatLog/Log"),
        helper.make_node(
            "Mul", ["Logged", "scale"], ["Y"], name="MathConcatLog/truediv"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "math_concat_log_fusion",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 7])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[max_value, add_value, scale_value],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_math_concat_log_fusion(tmp_path):
    model = _build_math_concat_log_model()
    x = np.linspace(-2.0, 4.0, num=3 * 7, dtype=np.float32).reshape(3, 7)
    feeds = {"X": x}

    (actual,) = run_model_and_compare(model, feeds, rtol=1e-6, atol=1e-6)
    expected = np.log(np.maximum(x, np.float32(1.0e-6)) + np.float32(1.0))
    expected = expected * np.float32(0.5)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "math_concat_log_fusion")
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
    assert not ({"Max", "Add", "Log", "Mul"} & op_names)
