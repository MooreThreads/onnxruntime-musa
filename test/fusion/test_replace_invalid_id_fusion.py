# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for the ReplaceInvalidId fusion."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _build_replace_invalid_id_model(
    dtype: np.dtype,
    threshold: int,
    replacement: int,
    reverse_mul_inputs: bool = False,
    detached_not_input: bool = False,
) -> bytes:
    dtype = np.dtype(dtype)
    elem_type = helper.np_dtype_to_tensor_dtype(dtype)
    threshold_initializer = numpy_helper.from_array(
        np.array(threshold, dtype=dtype), name="threshold"
    )
    replacement_initializer = numpy_helper.from_array(
        np.array(replacement, dtype=dtype), name="replacement"
    )
    keep_mul_inputs = ["Ids", "KeepMask"]
    replacement_mul_inputs = ["InvalidMask", "replacement"]
    if reverse_mul_inputs:
        keep_mul_inputs.reverse()
        replacement_mul_inputs.reverse()

    nodes = [
        helper.make_node(
            "LessOrEqual", ["Ids", "threshold"], ["IsInvalid"], name="LessEqual"
        ),
        helper.make_node(
            "Not",
            ["ExternalInvalid" if detached_not_input else "IsInvalid"],
            ["IsValid"],
            name="SelectV2__not",
        ),
        helper.make_node(
            "Cast", ["IsValid"], ["KeepMask"], name="SelectV2__keep_cast", to=elem_type
        ),
        helper.make_node("Mul", keep_mul_inputs, ["KeptIds"], name="SelectV2__keep"),
        helper.make_node(
            "Cast",
            ["IsInvalid"],
            ["InvalidMask"],
            name="SelectV2__invalid_cast",
            to=elem_type,
        ),
        helper.make_node(
            "Mul",
            replacement_mul_inputs,
            ["ReplacementIds"],
            name="SelectV2__replacement",
        ),
        helper.make_node(
            "Add", ["ReplacementIds", "KeptIds"], ["Y"], name="SelectV2"
        ),
    ]
    graph_inputs = [helper.make_tensor_value_info("Ids", elem_type, ["N", "M"])]
    if detached_not_input:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "ExternalInvalid", TensorProto.BOOL, ["N", "M"]
            )
        )
    graph = helper.make_graph(
        nodes,
        "replace_invalid_id_fusion",
        graph_inputs,
        [helper.make_tensor_value_info("Y", elem_type, None)],
        initializer=[threshold_initializer, replacement_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


@pytest.mark.parametrize(
    "ids,threshold,replacement,reverse_mul_inputs",
    [
        (
            np.array([[-5, 0, 1], [17, -1, 2]], dtype=np.int32),
            0,
            -1,
            False,
        ),
        (
            np.array(
                [
                    [2**53 + 5, 2**53 + 7],
                    [2**53 + 8, np.iinfo(np.int64).max],
                ],
                dtype=np.int64,
            ),
            2**53 + 7,
            np.iinfo(np.int64).min + 17,
            True,
        ),
        (np.empty((0, 3), dtype=np.int64), 0, -1, False),
    ],
)
def test_replace_invalid_id_accuracy(
    ids, threshold, replacement, reverse_mul_inputs
):
    model = _build_replace_invalid_id_model(
        ids.dtype, threshold, replacement, reverse_mul_inputs
    )
    (actual,) = run_model_and_compare(model, {"Ids": ids}, rtol=0, atol=0)
    expected = np.where(ids <= threshold, np.array(replacement, ids.dtype), ids)
    np.testing.assert_array_equal(actual, expected)


def test_replace_invalid_id_fusion_assignment(tmp_path):
    ids = np.array([[-2], [0], [1], [7]], dtype=np.int64)
    model = _build_replace_invalid_id_model(np.int64, 0, -1)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp_path / "replace_invalid_id_fusion")
    so.add_provider_for_devices(musa_devices(), {})
    session = ort.InferenceSession(model, sess_options=so)
    (actual,) = session.run(None, {"Ids": ids})
    np.testing.assert_array_equal(actual, np.array([[-1], [-1], [1], [7]]))
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)

    node_events = [
        event
        for event in events
        if event.get("cat") == "Node"
        and event.get("name", "").endswith("_kernel_time")
    ]
    op_names = {event.get("args", {}).get("op_name") for event in node_events}
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in op_names)
    assert not ({"LessOrEqual", "Not", "Cast", "Mul", "Add"} & op_names)


def test_replace_invalid_id_matcher_rejects_detached_not_input():
    ids = np.array([[-2, 0], [1, 7]], dtype=np.int64)
    external_invalid = np.array([[True, False], [False, True]])
    model = _build_replace_invalid_id_model(
        np.int64, 0, -1, detached_not_input=True
    )

    (actual,) = run_model_and_compare(
        model,
        {"Ids": ids, "ExternalInvalid": external_invalid},
        rtol=0,
        atol=0,
    )
    expected = np.where(ids <= 0, -1, 0) + np.where(~external_invalid, ids, 0)
    np.testing.assert_array_equal(actual, expected)
