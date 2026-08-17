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
"""End-to-end test for the masked modulo index + Gather fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    float32_to_bfloat16_bits,
    musa_devices,
    run_model_and_compare,
    run_with_iobinding,
)


def _build_modulo_gather_model() -> bytes:
    invalid = numpy_helper.from_array(np.array([0], dtype=np.int64), name="invalid")
    modulus = numpy_helper.from_array(np.array(5, dtype=np.int64), name="modulus")
    offset = numpy_helper.from_array(np.array(1, dtype=np.int64), name="offset")
    table = numpy_helper.from_array(
        np.arange(6 * 4, dtype=np.float32).reshape(6, 4), name="table"
    )
    nodes = [
        helper.make_node(
            "Equal", ["Ids", "invalid"], ["IsInvalid"], name="NotEqual_14"
        ),
        helper.make_node(
            "Not", ["IsInvalid"], ["IsValid"], name="NotEqual_14__6042"
        ),
        helper.make_node(
            "Cast", ["IsValid"], ["Mask"], to=TensorProto.INT64, name="Cast_14"
        ),
        helper.make_node("Div", ["Ids", "modulus"], ["DivOut"], name="Div__6044"),
        helper.make_node("Mul", ["DivOut", "modulus"], ["MulOut"], name="Mul__6046"),
        helper.make_node("Sub", ["Ids", "MulOut"], ["Remainder"], name="FloorMod"),
        helper.make_node("Add", ["Remainder", "offset"], ["Shifted"], name="add"),
        helper.make_node("Mul", ["Shifted", "Mask"], ["GatherIds"], name="mul_18"),
        helper.make_node(
            "Gather",
            ["table", "GatherIds"],
            ["Y"],
            axis=0,
            name="embedding_lookup_14",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "modulo_gather_fusion",
        [helper.make_tensor_value_info("Ids", TensorProto.INT64, ["N", 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[invalid, modulus, offset, table],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _build_modulo_gather_bfloat16_model(table_bits: np.ndarray) -> bytes:
    invalid = numpy_helper.from_array(np.array([0], dtype=np.int64), name="invalid")
    modulus = numpy_helper.from_array(np.array(5, dtype=np.int64), name="modulus")
    offset = numpy_helper.from_array(np.array(1, dtype=np.int64), name="offset")
    table = numpy_helper.from_array(table_bits, name="table")
    table.data_type = TensorProto.BFLOAT16
    nodes = [
        helper.make_node("Equal", ["Ids", "invalid"], ["IsInvalid"]),
        helper.make_node("Not", ["IsInvalid"], ["IsValid"]),
        helper.make_node("Cast", ["IsValid"], ["Mask"], to=TensorProto.INT64),
        helper.make_node("Div", ["Ids", "modulus"], ["DivOut"]),
        helper.make_node("Mul", ["DivOut", "modulus"], ["MulOut"]),
        helper.make_node("Sub", ["Ids", "MulOut"], ["Remainder"]),
        helper.make_node("Add", ["Remainder", "offset"], ["Shifted"]),
        helper.make_node("Mul", ["Shifted", "Mask"], ["GatherIds"]),
        helper.make_node("Gather", ["table", "GatherIds"], ["Y"], axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        "modulo_gather_bfloat16_fusion",
        [helper.make_tensor_value_info("Ids", TensorProto.INT64, ["N", 1])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, None)],
        initializer=[invalid, modulus, offset, table],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def test_modulo_gather_fusion(tmp_path):
    model = _build_modulo_gather_model()
    ids = np.array([[0], [1], [5], [6], [12]], dtype=np.int64)
    feeds = {"Ids": ids}

    (actual,) = run_model_and_compare(model, feeds, rtol=0, atol=0)
    table = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    gather_ids = np.where(ids == 0, 0, (ids - (ids // 5) * 5) + 1)
    expected = table[gather_ids]
    np.testing.assert_array_equal(actual, expected)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "modulo_gather_fusion")
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
    assert not ({"Equal", "Not", "Cast", "Div", "Sub", "Add", "Gather"} & op_names)


def test_modulo_gather_fusion_bfloat16_payload(tmp_path):
    table_bits = float32_to_bfloat16_bits(
        np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    )
    model = _build_modulo_gather_bfloat16_model(table_bits)
    ids = np.array([[0], [1], [5], [6], [12]], dtype=np.int64)
    feeds = {"Ids": ids}

    (actual,) = run_with_iobinding(
        model,
        feeds,
        {},
        [("Y", TensorProto.BFLOAT16, (5, 1, 4))],
        use_musa=True,
    )
    gather_ids = np.where(ids == 0, 0, (ids - (ids // 5) * 5) + 1)
    expected = table_bits[gather_ids]
    np.testing.assert_array_equal(actual, expected)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "modulo_gather_fusion_bf16")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    io_binding = session.io_binding()
    io_binding.bind_input(
        "Ids", "cpu", 0, TensorProto.INT64, ids.shape, ids.ctypes.data
    )
    output = np.empty((5, 1, 4), dtype=np.uint16)
    io_binding.bind_output(
        "Y", "cpu", 0, TensorProto.BFLOAT16, output.shape, output.ctypes.data
    )
    session.run_with_iobinding(io_binding)
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
