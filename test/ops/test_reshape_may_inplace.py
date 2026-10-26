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
"""Reshape may-inplace: reuse the input buffer only when the allocator allows.

c4 removed the hard input/output alias so a CPU-produced input could never be
read as device memory. With MemcpyFromHost registered the Reshape kernel only
receives device tensors, so the kernel can advertise a *mutable* alias: ORT
reuses the buffer when it is safe, the kernel skips the copy when src == dst,
and every other case still copies (c4's safety is not weakened).
"""

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from op_test_utils import (
    build_graph_model,
    musa_devices,
    run_model_and_compare_with_cpu_fallback,
)


def _musa_session(model: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(musa_devices(), {})
    return ort.InferenceSession(model, sess_options=options)


def test_reshape_may_inplace_reuses_single_consumer_device_output(
    capfd, monkeypatch
):
    """Add's device output feeds only Reshape: the buffer can be reused."""
    monkeypatch.setenv("ORT_MUSA_RESHAPE_TRACE", "1")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    shape = np.array([3, 2], dtype=np.int64)
    one = np.array(1.0, dtype=np.float32)
    model = build_graph_model(
        [
            helper.make_node("Add", ["X", "one"], ["added"]),
            helper.make_node("Reshape", ["added", "shape"], ["reshaped"]),
            # Keep the Reshape output internal: a graph output does not share
            # a buffer with the input, so the alias cannot be observed there.
            helper.make_node("Relu", ["reshaped"], ["Y"]),
        ],
        {"X": x, "shape": shape},
        [("Y", TensorProto.FLOAT)],
        initializers=[numpy_helper.from_array(one, name="one")],
        name="reshape_may_inplace_reuse_graph",
    )

    capfd.readouterr()
    session = _musa_session(model)
    out = session.run(None, {"X": x, "shape": shape})[0]
    err = capfd.readouterr().err
    np.testing.assert_allclose(
        out, np.maximum((x + 1.0).reshape(3, 2), 0.0), rtol=0, atol=0
    )
    assert "MUSA_RESHAPE inplace=1" in err, err


def test_reshape_may_inplace_copies_with_another_consumer(
    capfd, monkeypatch
):
    """Add's output has a second consumer, so Reshape must copy."""
    monkeypatch.setenv("ORT_MUSA_RESHAPE_TRACE", "1")
    rng = np.random.default_rng(8)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    shape = np.array([3, 2], dtype=np.int64)
    one = np.array(1.0, dtype=np.float32)
    model = build_graph_model(
        [
            helper.make_node("Add", ["X", "one"], ["added"]),
            helper.make_node("Reshape", ["added", "shape"], ["Y"]),
            helper.make_node("Relu", ["added"], ["Z"]),
        ],
        {"X": x, "shape": shape},
        [("Y", TensorProto.FLOAT), ("Z", TensorProto.FLOAT)],
        initializers=[numpy_helper.from_array(one, name="one")],
        name="reshape_may_inplace_copy_graph",
    )

    capfd.readouterr()
    session = _musa_session(model)
    y, z = session.run(None, {"X": x, "shape": shape})
    err = capfd.readouterr().err
    np.testing.assert_allclose(y, (x + 1.0).reshape(3, 2), rtol=0, atol=0)
    np.testing.assert_allclose(z, np.maximum(x + 1.0, 0.0), rtol=0, atol=0)
    assert "MUSA_RESHAPE inplace=0" in err, err


def test_reshape_after_cpu_produced_input_stays_correct(monkeypatch):
    """A CPU-produced input is staged; the may-inplace alias stays safe."""
    x = np.random.default_rng(10).standard_normal((1, 4, 1029)).astype(np.float16)
    shape = np.array([1, 4, 1029], dtype=np.int64)
    model = build_graph_model(
        [
            helper.make_node("Sin", ["X"], ["cpu_output"]),
            helper.make_node("Reshape", ["cpu_output", "shape"], ["Y"]),
        ],
        {"X": x, "shape": shape},
        [("Y", TensorProto.FLOAT16)],
        name="reshape_may_inplace_staged_graph",
    )
    monkeypatch.setenv("ORT_MUSA_TEST_FAIL_RESHAPE_CPU_INPUT", "1")
    run_model_and_compare_with_cpu_fallback(
        model, {"X": x, "shape": shape}, disable_ep_error_fallback=True
    )
