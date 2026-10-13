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
"""CPU/device boundary copies must be ordered on the EP's compute stream.

A small tensor produced on the CPU side (a `Shape`/`Slice` metadata chain) that is
consumed by a device kernel is transferred by the plugin's MemcpyFromHost kernel.
That transfer must be issued on the caller's compute stream: a blocking
musaMemcpy() on the null stream is ordered against neither the kernels that are
still running on the compute stream (ORT's memory pattern may have handed the
destination block of a tensor whose last consumer just ran) nor the consumers
that are enqueued after it.

Regression cover for sherpa_whisper_base / zipformer_encoder_en /
campplus_3dspeaker / campplus_cosyvoice: with ORT graph optimisation on, the
transferred metadata was read back as the previous owner's bytes, so the shape
chain produced garbage sizes (`Add`/`Mul` broadcast mismatch, `Reshape`
"invalid argument", NaNs). Every case therefore runs the graph twice inside one
session - the first run allocates dynamically, the second reuses the memory
pattern - and compares both runs against the CPU EP.
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
    return ort.InferenceSession(model_bytes, sess_options=options,
                                providers=["CPUExecutionProvider"])


def _metadata_boundary_copy_model(copy_bytes, tiny_writers):
    """`Mul` (device) -> `Shape` -> `Slice` (CPU) -> `Cast` (device) -> `Mul`.

    The `Cast` node runs on the device and consumes CPU-side metadata, so ORT's
    MemcpyTransformer has to insert a MemcpyFromHost for `copy_bytes` bytes. The
    `tiny_writers` device tensors of the same size that are produced and dropped
    right before the copy make ORT's memory pattern hand the copy destination a
    block that a kernel which is still running on the device wrote.
    """
    rng = np.random.RandomState(0)
    x = rng.standard_normal((1, 1500, 512)).astype(np.float32)
    nodes = []
    initializers = [
        helper.make_tensor("c", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("ax0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("ax012", TensorProto.INT64, [3], [0, 1, 2]),
        helper.make_tensor("s0", TensorProto.INT64, [1], [0]),
    ]

    for index in range(tiny_writers):
        # A reduction over the whole activation: enqueued right before the
        # boundary copy, still running device-side when the copy is issued.
        if copy_bytes == 4:
            # float32 scalar -> 4 bytes
            nodes.append(helper.make_node("ReduceSum", ["X", "ax012"], [f"t{index}"],
                                          keepdims=1))
            nodes.append(helper.make_node("Mul", [f"t{index}", "c"], [f"u{index}"]))
        else:
            # int64 element -> 8 bytes (dtype and size of the copied metadata)
            nodes.append(helper.make_node("ReduceSum", ["X", "ax012"], [f"t{index}"],
                                          keepdims=0))
            nodes.append(helper.make_node("Cast", [f"t{index}"], [f"t{index}i"],
                                          to=TensorProto.INT64))
            nodes.append(helper.make_node("Mul", [f"t{index}i", "ax0"], [f"u{index}"]))

    nodes.append(helper.make_node("Shape", ["X"], ["sc"]))          # CPU, rank 3
    if copy_bytes == 8:
        initializers.append(helper.make_tensor("s1", TensorProto.INT64, [1], [1]))
        nodes.append(helper.make_node("Slice", ["sc", "s0", "s1", "ax0"], ["meta"]))
    elif copy_bytes == 4:
        # CPU-side int32 slice: 4 bytes cross the boundary
        initializers.append(helper.make_tensor("s1", TensorProto.INT64, [1], [1]))
        nodes.append(helper.make_node("Slice", ["sc", "s0", "s1", "ax0"], ["slice0"]))
        nodes.append(helper.make_node("Cast", ["slice0"], ["meta"], to=TensorProto.INT32))
    else:
        raise ValueError(copy_bytes)
    nodes.append(helper.make_node("Cast", ["meta"], ["vf"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["X", "vf"], ["Y"]))

    model = build_graph_model(
        nodes,
        inputs={"X": x},
        outputs=[("vf", TensorProto.FLOAT), ("Y", TensorProto.FLOAT)],
        initializers=initializers,
        opset=17,
        name=f"boundary_copy_{copy_bytes}_{tiny_writers}",
    )
    return model, {"X": x}


@pytest.mark.parametrize("copy_bytes,tiny_writers", [(4, 1), (4, 4), (8, 1), (8, 4)])
def test_small_boundary_copy_stays_ordered(copy_bytes, tiny_writers):
    model, feeds = _metadata_boundary_copy_model(copy_bytes, tiny_writers)
    cpu_vf, cpu_y = _session(model, musa=False).run(None, dict(feeds))
    session = _session(model, musa=True)
    for run_index in range(_RUNS):
        vf, y = session.run(None, dict(feeds))
        np.testing.assert_array_equal(np.asarray(vf), np.asarray(cpu_vf),
                                      err_msg=f"run {run_index}: vf")
        np.testing.assert_allclose(np.asarray(y), np.asarray(cpu_y), rtol=1e-4, atol=1e-3,
                                   err_msg=f"run {run_index}: Y")
        assert np.isfinite(np.asarray(y)).all(), f"run {run_index}: Y has NaN/Inf"
