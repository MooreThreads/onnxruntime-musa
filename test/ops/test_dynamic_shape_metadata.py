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
"""Dynamic-shape metadata chains that feed Slice.

`Shape`/`Gather`/`Unsqueeze` arithmetic produces the runtime shapes of a graph.
Where that arithmetic derives from a device tensor it must not be executed as a
device-side subgraph: the values belong on the CPU EP, and most operators that
consume them (`Reshape` shape, `Tile` repeats, ...) declare those inputs as
`OrtMemTypeCPUInput`, exactly like the in-tree CUDA kernels do. `Slice` no
longer declares starts/ends/axes/steps; its kernel reads them through
`ReadIntTensors` on the compute stream, so a device-resident metadata chain is
read correctly without ORT-inserting a host copy for every execution.

Regression cover for sherpa_whisper_base / campplus_3dspeaker and
zipformer_encoder_en: with ORT graph optimisation on, their `Slice` received
`ends=[0]` because the `Shape -> Gather -> Unsqueeze -> Slice` chain ran on the
MUSA EP and ORT's memory pattern later reused the tiny device buffers of that
chain for multi-MB activations. Running the model twice inside one session is
what exposes it (the first run allocates dynamically, the second reuses the
pattern), so every case below is executed twice per provider.
"""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from op_test_utils import build_graph_model, musa_devices

_RUNS = 2


def _session(model_bytes, *, musa):
    options = ort.SessionOptions()
    if musa:
        devices = musa_devices()
        if not devices:
            raise RuntimeError("No MUSA device available; refusing a CPU-only session")
        options.add_provider_for_devices(devices, {})
        session = ort.InferenceSession(model_bytes, sess_options=options)
        # Never let ORT silently rebuild the session CPU-only after an EP failure.
        session.disable_fallback()
        return session
    return ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])


def _run_twice(session, feeds):
    return [session.run(None, dict(feeds)) for _ in range(_RUNS)]


def _slice_ends_chain_model(permute_shape_chain):
    """`Add` (device) -> `Shape` -> `Gather` -> `Unsqueeze` -> `Slice(ends=..)`.

    `pos` is [1500, 512] and the `ends` value is the runtime length of the first
    axis of `x`, so the slice is a no-op: the output must equal `pos` exactly on
    every run. Anything else means the metadata lost its value.
    """
    rng = np.random.RandomState(0)
    # The permuted variant mirrors ORT's TransposeOptimizer rewrite of
    # `Shape(Transpose(x))`: the Shape reads the *pre*-transpose tensor, so the
    # device tensor is [1, 512, 1500] and the permuted gather yields [1, 1500, 512].
    x = rng.standard_normal((1, 512, 1500) if permute_shape_chain else (1, 1500, 512)).astype(np.float32)
    pos = rng.standard_normal((1500, 512)).astype(np.float32)

    nodes = [helper.make_node("Add", ["X", "zero"], ["dev"]),
             helper.make_node("Shape", ["dev"], ["shape"])]
    if permute_shape_chain:
        # Mirrors ORT's TransposeOptimizer rewrite of `Shape(Transpose(x))`.
        nodes.append(helper.make_node("Gather", ["shape", "perm"], ["shape_perm"], axis=0))
        nodes.append(helper.make_node("Gather", ["shape_perm", "one"], ["dim"], axis=0))
    else:
        nodes.append(helper.make_node("Gather", ["shape", "one"], ["dim"], axis=0))
    nodes.append(helper.make_node("Unsqueeze", ["dim", "axis0"], ["ends"]))
    nodes.append(helper.make_node("Slice", ["pos", "zero_vec", "ends", "axis0", "step1"], ["Y"]))

    initializers = [
        helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
        helper.make_tensor("perm", TensorProto.INT64, [3], [0, 2, 1]),
        helper.make_tensor("one", TensorProto.INT64, [], [1]),
        helper.make_tensor("axis0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("zero_vec", TensorProto.INT64, [1], [0]),
        helper.make_tensor("step1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("pos", TensorProto.FLOAT, [1500, 512], pos.ravel().tolist()),
    ]
    model = build_graph_model(
        nodes,
        inputs={"X": x},
        outputs=[("ends", TensorProto.INT64), ("Y", TensorProto.FLOAT)],
        initializers=initializers,
        opset=17,
        name="metadata_slice_chain_graph",
    )
    return model, {"X": x}, pos


@pytest.mark.parametrize("permute_shape_chain", [False, True])
def test_slice_ends_from_device_shape_chain(permute_shape_chain):
    model, feeds, pos = _slice_ends_chain_model(permute_shape_chain)
    musa_runs = _run_twice(_session(model, musa=True), feeds)
    cpu_ends, cpu_y = _run_twice(_session(model, musa=False), feeds)[0]
    for run_index, (ends, y) in enumerate(musa_runs):
        np.testing.assert_array_equal(
            np.asarray(ends), np.asarray(cpu_ends),
            err_msg=f"ends mismatch on run {run_index}")
        np.testing.assert_array_equal(np.asarray(ends), np.array([1500], dtype=np.int64))
        assert np.asarray(y).shape == (1500, 512), f"run {run_index}: {np.asarray(y).shape}"
        np.testing.assert_allclose(np.asarray(y), pos, rtol=0, atol=0)


@pytest.mark.parametrize("permute_shape_chain", [False, True])
def test_slice_ends_metadata_chain_stays_off_the_device_copy_route(permute_shape_chain, monkeypatch):
    """The chain must not need a MUSA CPU/device boundary copy.

    `ORT_MUSA_TEST_FAIL_MEMCPY_KERNEL` makes the plugin's MemcpyFromHost /
    MemcpyToHost kernels fail. Metadata that is handled on the CPU EP needs no
    such boundary copy, so the run must still succeed with the hook armed; if
    the chain drifts back onto the device kernels the test fails.
    """
    model, feeds, pos = _slice_ends_chain_model(permute_shape_chain)
    monkeypatch.setenv("ORT_MUSA_TEST_FAIL_MEMCPY_KERNEL", "1")
    for run_index, (ends, y) in enumerate(_run_twice(_session(model, musa=True), feeds)):
        np.testing.assert_array_equal(np.asarray(ends), np.array([1500], dtype=np.int64),
                                      err_msg=f"run {run_index}")
        np.testing.assert_allclose(np.asarray(y), pos, rtol=0, atol=0)
