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
"""Optional attributes that the graph layer has to read as "not present".

`OrtApi::Node_GetAttributeByName` reports an attribute it cannot provide - an
optional attribute with no constant default value, such as Transpose's `perm`
or ReduceMean's `axes` - as a success status with an *empty* handle. Every
attribute reader in `graph/graph_utils.cc` (and the copies of it under
`fusion/`) used to call into that handle anyway, which dereferences null inside
`OpAttr_GetType` and takes the whole process down while the EP is answering
GetCapability.

Each graph below is built so that a fusion matcher really does reach one of
those reads, which is what makes the case discriminating: before the readers
checked the handle, building the session crashed; afterwards the matcher sees
"attribute absent", declines the fusion, and the graph runs as plain ONNX ops
on the MUSA EP with the same numbers as the CPU EP.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from op_test_utils import build_graph_model, musa_devices, run


def _require_devices():
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for the attribute default tests")
    return devices


def _split_concat_transpose_model():
    """Split -> Concat -> Transpose with no `perm` on the Transpose.

    `FindSplitConcatFusions` walks exactly this shape: a rank-3 input split
    along axis 2 into equal parts, concatenated along axis 0, then a Transpose
    whose `perm` it compares with {0, 2, 1} - the read that used to crash. With
    no `perm` the Transpose means "reverse the dimensions".
    """
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    model_bytes = build_graph_model(
        [
            helper.make_node("Split", ["X", "split_sizes"], ["s0", "s1"], axis=2),
            helper.make_node("Concat", ["s0", "s1"], ["concat_out"], axis=0),
            helper.make_node("Transpose", ["concat_out"], ["Y"]),
        ],
        {"X": x},
        [("Y", TensorProto.FLOAT)],
        initializers=[
            helper.make_tensor("split_sizes", TensorProto.INT64, [2], [2, 2]),
        ],
        opset=17,
        name="split_concat_transpose_without_perm",
    )
    return model_bytes, {"X": x}


def _rms_norm_without_axes_model():
    """The RMSNorm decomposition with no `axes` on the ReduceMean.

    `FindRmsNormFusions` / `FindCastRmsNormFusions` walk
    Mul -> ReduceMean -> Add -> Sqrt -> Div -> Mul and read the reduce node's
    `axes` to check that it reduces the last dimension - the read that used to
    crash. A ReduceMean without `axes` reduces *every* dimension, which is not
    what those fusions describe, so the matchers must decline; the graph then
    runs as written. Every leading dimension is 1 so that the reduce output
    keeps the shape the matcher checks before it reads the attribute.
    """
    x = np.arange(1 * 1 * 4, dtype=np.float32).reshape(1, 1, 4) / 4.0
    model_bytes = build_graph_model(
        [
            helper.make_node("Mul", ["X", "X"], ["square"]),
            helper.make_node("ReduceMean", ["square"], ["mean"]),
            helper.make_node("Add", ["mean", "epsilon"], ["var"]),
            helper.make_node("Sqrt", ["var"], ["std"]),
            helper.make_node("Div", ["X", "std"], ["normed"]),
            helper.make_node("Mul", ["normed", "gamma"], ["Y"]),
        ],
        {"X": x},
        [("Y", TensorProto.FLOAT)],
        initializers=[
            helper.make_tensor("epsilon", TensorProto.FLOAT, [], [1e-5]),
            helper.make_tensor("gamma", TensorProto.FLOAT, [4], [1.0, 1.0, 1.0, 1.0]),
        ],
        opset=17,
        name="rms_norm_without_axes",
    )
    return model_bytes, {"X": x}


@pytest.mark.parametrize(
    "builder,label",
    [(_split_concat_transpose_model, "split-concat-transpose-without-perm"),
     (_rms_norm_without_axes_model, "rms-norm-without-axes")],
    ids=["transpose-without-perm", "reduce-mean-without-axes"],
)
def test_optional_attribute_without_default(capfd, builder, label):
    model_bytes, feeds = builder()
    devices = _require_devices()
    cpu_outputs = run(model_bytes, feeds, use_musa=False)

    options = ort.SessionOptions()
    # An op the EP does not claim must fail loudly instead of running on the
    # CPU EP, so the numbers below are really the MUSA ones.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {"dtype_diagnostics": "1"})
    capfd.readouterr()
    session = ort.InferenceSession(model_bytes, sess_options=options)
    musa_outputs = session.run(None, dict(feeds))

    np.testing.assert_allclose(
        musa_outputs[0], cpu_outputs[0], rtol=1e-5, atol=1e-6,
        err_msg=f"{label}: CPU vs MUSA mismatch",
    )
    node_lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("MUSA_DTYPE_NODE")
    ]
    assert node_lines, f"{label}: the EP reported no nodes"
    for line in node_lines:
        assert line.rstrip().endswith("assigned=1"), f"{label}: {line}"
