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
"""Tile repeats computed from a shape-only chain, absorbed into Tile+Concat.

A fused node cannot declare a CPU input, so ORT copies a runtime repeats tensor
onto the device and the kernel would have to read it back with a stream
synchronize. Instead the matcher absorbs the Shape/Gather/... chain and the
kernel recomputes repeats from the shapes of its inputs. These tests cover the
cases that make that absorption observable: a repeats value that follows
another tensor's shape, a chain value with a consumer outside the fusion, and
inputs the evaluator has to reject.
"""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from op_test_utils import musa_devices

INT64_MAX = 2**63 - 1


def _build_model(name, nodes, inputs, outputs, initializers, opset=17):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _int64_initializer(name, values, shape=None):
    array = np.asarray(values, dtype=np.int64)
    if shape is not None:
        array = array.reshape(shape)
    return helper.make_tensor(
        name, TensorProto.INT64, list(array.shape), array.reshape(-1).tolist()
    )


def _float_initializer(name, values, shape):
    array = np.asarray(values, dtype=np.float32).reshape(shape)
    return helper.make_tensor(
        name, TensorProto.FLOAT, list(array.shape), array.reshape(-1).tolist()
    )


def _session(model, use_musa, profile_prefix=None):
    if not use_musa:
        return ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    devices = musa_devices()
    assert devices, "no MUSA device: refusing a CPU-only session for the MUSA run"
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    options.add_provider_for_devices(devices, {})
    return ort.InferenceSession(model, sess_options=options)


def _read_ops_by_provider(session):
    profile_path = session.end_profiling()
    try:
        with open(profile_path, "r", encoding="utf-8") as handle:
            events = json.load(handle)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)
    ops = {}
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith(
            "_kernel_time"
        ):
            continue
        args = event.get("args", {})
        ops.setdefault(args.get("provider"), set()).add(args.get("op_name"))
    return ops


def _musa_ops(ops):
    return ops.get("MUSAExecutionProvider", set())


def _fused_ops(ops):
    return {
        op for op in _musa_ops(ops) if str(op).startswith("MUSAExecutionProvider_")
    }


def _assert_tile_concat_absorbed(ops):
    """The Tile and Concat ran inside the fused node, chain and all."""
    assert _fused_ops(ops)
    assert "Tile" not in _musa_ops(ops)
    assert "Concat" not in _musa_ops(ops)


def _batch_chain_nodes(repeats_source):
    return [
        helper.make_node("Shape", ["query"], ["query_shape"]),
        helper.make_node("Gather", ["query_shape", "zero"], ["batch"], axis=0),
        *repeats_source,
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y"], axis=1),
    ]


def _batch_chain_initializers():
    return [
        _int64_initializer("zero", [0]),
        _int64_initializer("one", [1]),
        _float_initializer("data", np.arange(4, dtype=np.float32), (1, 4)),
    ]


def test_tile_concat_repeats_follows_another_tensors_shape(tmp_path):
    """One session, two batch sizes: repeats moves while the Tile input does not.

    The Tile data stays [1, 4] and only `query`'s batch axis changes. The matcher
    absorbs the chain, so repeats is no longer an input of the fused node and the
    expression is the only way to obtain it.
    """
    rng = np.random.default_rng(29)
    nodes = _batch_chain_nodes(
        [
            helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
            helper.make_node("Concat", ["batch_1d", "one"], ["repeats"], axis=0),
        ]
    )
    model = _build_model(
        "tile_concat_repeats_shape_chain",
        nodes,
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        _batch_chain_initializers(),
    )

    cpu = _session(model, use_musa=False)
    musa = _session(model, use_musa=True, profile_prefix=str(tmp_path / "shape_chain"))
    for batch in (1, 3):
        query = rng.standard_normal((batch, 4)).astype(np.float32)
        expected = cpu.run(None, {"query": query})[0]
        actual = musa.run(None, {"query": query})[0]
        assert actual.shape == expected.shape == (batch, 8)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    _assert_tile_concat_absorbed(_read_ops_by_provider(musa))


def test_tile_concat_repeats_chain_value_used_outside_fusion():
    """A chain value that is also a graph output must not be absorbed.

    A fused node owns its internal values, so absorbing a value another consumer
    reads would either break that consumer or change the graph. The Tile falls
    back to the standalone kernel (which takes repeats as a CPU input), the chain
    stays in the graph, and both outputs stay correct.
    """
    rng = np.random.default_rng(31)
    nodes = _batch_chain_nodes(
        [
            helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
            helper.make_node("Concat", ["batch_1d", "one"], ["repeats"], axis=0),
        ]
    )
    model = _build_model(
        "tile_concat_repeats_chain_external_consumer",
        nodes,
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, ["batch", 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("batch", TensorProto.INT64, None),
        ],
        _batch_chain_initializers(),
    )

    query = rng.standard_normal((2, 4)).astype(np.float32)
    feeds = {"query": query}
    expected = _session(model, use_musa=False).run(None, feeds)
    actual = _session(model, use_musa=True).run(None, feeds)
    assert len(actual) == 2
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[1] == np.int64(2)


def test_tile_concat_repeats_chain_with_negative_slice():
    """Negative Slice bounds are resolved against the runtime shape."""
    rng = np.random.default_rng(37)
    nodes = [
        helper.make_node("Shape", ["query"], ["query_shape"]),
        helper.make_node(
            "Slice", ["query_shape", "head_start", "head_end"], ["batch_dim"]
        ),
        helper.make_node("Concat", ["batch_dim", "one"], ["repeats"], axis=0),
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y"], axis=1),
    ]
    initializers = [
        _float_initializer("data", np.arange(4, dtype=np.float32), (1, 4)),
        _int64_initializer("one", [1]),
        _int64_initializer("head_start", [-2]),
        _int64_initializer("head_end", [-1]),
    ]
    model = _build_model(
        "tile_concat_repeats_negative_slice",
        nodes,
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )

    for batch in (2, 5):
        query = rng.standard_normal((batch, 4)).astype(np.float32)
        feeds = {"query": query}
        expected = _session(model, use_musa=False).run(None, feeds)[0]
        actual = _session(model, use_musa=True).run(None, feeds)[0]
        assert actual.shape == expected.shape == (batch, 8)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_tile_concat_repeats_chain_with_reshape_element_mismatch():
    """An unusable shape value fails loudly instead of running on stale data."""
    nodes = [
        helper.make_node("Shape", ["query"], ["query_shape"]),
        helper.make_node("Reshape", ["query_shape", "three"], ["bad_repeats"]),
        helper.make_node("Tile", ["data", "bad_repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y"], axis=1),
    ]
    initializers = [
        _float_initializer("data", np.arange(2, dtype=np.float32), (1, 2)),
        _int64_initializer("three", [3]),
    ]
    model = _build_model(
        "tile_concat_repeats_reshape_mismatch",
        nodes,
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )

    feeds = {"query": np.zeros((1, 4), dtype=np.float32)}
    # Shape(query) holds two entries and the Reshape asks for three. The graph is
    # valid ONNX, so the failure has to surface at run time, not as a crash.
    with pytest.raises(Exception):
        _session(model, use_musa=False).run(None, feeds)
    with pytest.raises(Exception):
        _session(model, use_musa=True).run(None, feeds)
