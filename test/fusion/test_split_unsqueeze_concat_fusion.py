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
"""End-to-end tests for Reshape -> Split -> Unsqueeze -> Concat fusion."""

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
    session_options.profile_file_prefix = "split_unsqueeze_concat_fusion"
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


def _build_model(
    part_count, part_width, sequence, with_transpose, explicit_split, include_reshape
):
    packed_width = part_count * part_width
    initializers = []
    split_inputs = ["R" if include_reshape else "X"]
    nodes = []
    value_info = []
    input_shape = (
        ["batch", sequence * packed_width]
        if include_reshape
        else ["batch", sequence, packed_width]
    )
    if include_reshape:
        target_shape = helper.make_tensor(
            "target_shape", TensorProto.INT64, [3], [0, sequence, packed_width]
        )
        initializers.append(target_shape)
        nodes.append(
            helper.make_node(
                "Reshape", ["X", "target_shape"], ["R"], name="LateReshape"
            )
        )
        value_info.append(
            helper.make_tensor_value_info(
                "R", TensorProto.FLOAT, ["batch", sequence, packed_width]
            )
        )

    if explicit_split:
        split_sizes = helper.make_tensor(
            "split_sizes",
            TensorProto.INT64,
            [part_count],
            [part_width] * part_count,
        )
        initializers.append(split_sizes)
        split_inputs.append("split_sizes")

    split_outputs = [f"S{i}" for i in range(part_count)]
    unsqueeze_outputs = [f"U{i}" for i in range(part_count)]
    nodes.append(
        helper.make_node(
            "Split",
            split_inputs,
            split_outputs,
            axis=2,
            name="LateSplit",
        )
    )
    for i, split_output in enumerate(split_outputs):
        axes = helper.make_tensor(f"axes{i}", TensorProto.INT64, [1], [0])
        initializers.append(axes)
        nodes.append(
            helper.make_node(
                "Unsqueeze",
                [split_output, f"axes{i}"],
                [unsqueeze_outputs[i]],
                name=f"LateUnsqueeze{i}",
            )
        )
        value_info.append(
            helper.make_tensor_value_info(
                split_output, TensorProto.FLOAT, ["batch", sequence, part_width]
            )
        )
        value_info.append(
            helper.make_tensor_value_info(
                unsqueeze_outputs[i],
                TensorProto.FLOAT,
                [1, "batch", sequence, part_width],
            )
        )

    concat_output = "C" if with_transpose else "Y"
    nodes.append(
        helper.make_node(
            "Concat",
            unsqueeze_outputs,
            [concat_output],
            axis=0,
            name="LateConcat",
        )
    )
    concat_shape = [part_count, "batch", sequence, part_width]
    if with_transpose:
        value_info.append(
            helper.make_tensor_value_info("C", TensorProto.FLOAT, concat_shape)
        )
        nodes.append(
            helper.make_node(
                "Transpose",
                ["C"],
                ["Y"],
                perm=[0, 1, 3, 2],
                name="LateTranspose",
            )
        )
        output_shape = [part_count, "batch", part_width, sequence]
    else:
        output_shape = concat_shape

    graph = helper.make_graph(
        nodes,
        f"split_unsqueeze_concat_fusion_p{part_count}_graph",
        [
            helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, input_shape
            )
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


@pytest.mark.parametrize("with_transpose", [False, True])
@pytest.mark.parametrize(
    "part_count,part_width,explicit_split,include_reshape",
    [
        (2, 5, False, True),
        (2, 5, False, False),
        (3, 4, True, True),
    ],
)
def test_split_unsqueeze_concat_fusion_axis2_to_axis0(
    part_count, part_width, explicit_split, include_reshape, with_transpose
):
    rng = np.random.default_rng(41)
    batch = 3
    sequence = 4
    packed_width = part_count * part_width
    input_shape = (
        (batch, sequence * packed_width)
        if include_reshape
        else (batch, sequence, packed_width)
    )
    x = rng.standard_normal(input_shape).astype(np.float32)
    model_bytes = _build_model(
        part_count,
        part_width,
        sequence,
        with_transpose,
        explicit_split,
        include_reshape,
    )

    musa_outputs = run_model_and_compare(
        model_bytes,
        {"X": x},
        rtol=1e-6,
        atol=1e-6,
    )
    reshaped = x.reshape(batch, sequence, packed_width)
    pieces = np.split(reshaped, part_count, axis=2)
    expected = np.concatenate([np.expand_dims(piece, axis=0) for piece in pieces], axis=0)
    if with_transpose:
        expected = expected.transpose(0, 1, 3, 2)
    np.testing.assert_allclose(musa_outputs[0], expected, rtol=0, atol=0)

    node_names = _profile_musa_node_names(model_bytes, {"X": x})
    if include_reshape:
        assert not any(
            name.startswith("LateReshape") for name in node_names
        ), node_names
    assert not any(name.startswith("LateSplit") for name in node_names), node_names
    assert not any(name.startswith("LateUnsqueeze") for name in node_names), node_names
    assert not any(name.startswith("LateConcat") for name in node_names), node_names
    if with_transpose:
        assert not any(
            name.startswith("LateTranspose") for name in node_names
        ), node_names
