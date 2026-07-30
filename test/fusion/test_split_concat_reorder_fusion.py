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
"""End-to-end tests for Split(axis=2) -> Concat(axis=0) fusion."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from op_test_utils import TensorProto, musa_devices, run_model_and_compare


def _profile_musa_node_names(model_bytes, feeds):
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for profiling")

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = "split_concat_reorder_fusion"
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


@pytest.mark.parametrize(
    "part_count,part_width",
    [
        (2, 2),
        (4, 2),
        (8, 2),
        (4, 3),
        (4, 64),
    ],
)
def test_split_concat_reorder_fusion_rank3_axis2_to_axis0(part_count, part_width):
    rng = np.random.default_rng(37)
    sequence = 3
    packed_width = part_count * part_width
    x = rng.standard_normal((2, sequence * packed_width)).astype(np.float32)

    target_shape = helper.make_tensor(
        "target_shape", TensorProto.INT64, [3], [0, sequence, packed_width]
    )
    split_outputs = [f"S{i}" for i in range(part_count)]
    split_value_infos = [
        helper.make_tensor_value_info(
            name, TensorProto.FLOAT, ["batch", sequence, part_width]
        )
        for name in split_outputs
    ]
    nodes = [
        helper.make_node(
            "Reshape", ["X", "target_shape"], ["R"], name="ReorderReshape"
        ),
        helper.make_node(
            "Split",
            ["R"],
            split_outputs,
            axis=2,
            name="ReorderSplit",
        ),
        helper.make_node(
            "Concat",
            split_outputs,
            ["Y"],
            axis=0,
            name="ReorderConcat",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        f"split_concat_reorder_fusion_p{part_count}_graph",
        [
            helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, ["batch", sequence * packed_width]
            )
        ],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [f"batch{part_count}", sequence, part_width]
            )
        ],
        initializer=[target_shape],
        value_info=[
            helper.make_tensor_value_info(
                "R", TensorProto.FLOAT, ["batch", sequence, packed_width]
            ),
            *split_value_infos,
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    model_bytes = model.SerializeToString()

    musa_outputs = run_model_and_compare(
        model_bytes,
        {"X": x},
        rtol=1e-6,
        atol=1e-6,
    )
    expected = np.concatenate(
        np.split(x.reshape(2, sequence, packed_width), part_count, axis=2),
        axis=0,
    )
    np.testing.assert_allclose(musa_outputs[0], expected, rtol=0, atol=0)

    node_names = _profile_musa_node_names(model_bytes, {"X": x})
    assert not any(
        name.startswith("ReorderSplit") for name in node_names
    ), node_names
    assert not any(
        name.startswith("ReorderConcat") for name in node_names
    ), node_names


def test_split_concat_fusion_absorbs_optional_transpose():
    rng = np.random.default_rng(83)
    x = rng.standard_normal((2, 3, 16)).astype(np.float32)
    split_outputs = [f"S{i}" for i in range(4)]
    graph = helper.make_graph(
        [
            helper.make_node("Split", ["X"], split_outputs, axis=2,
                             name="SplitConcatSplit"),
            helper.make_node("Concat", split_outputs, ["C"], axis=0,
                             name="SplitConcatConcat"),
            helper.make_node("Transpose", ["C"], ["Y"], perm=[0, 2, 1],
                             name="SplitConcatTranspose"),
        ],
        "split_concat_optional_transpose_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 16])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [8, 4, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    model_bytes = model.SerializeToString()

    musa_outputs = run_model_and_compare(model_bytes, {"X": x}, rtol=1e-6, atol=1e-6)
    expected = np.transpose(np.concatenate(np.split(x, 4, axis=2), axis=0), (0, 2, 1))
    np.testing.assert_allclose(musa_outputs[0], expected, rtol=0, atol=0)

    node_names = _profile_musa_node_names(model_bytes, {"X": x})
    for name in ("SplitConcatSplit", "SplitConcatConcat", "SplitConcatTranspose"):
        assert not any(event.startswith(name) for event in node_names), node_names
