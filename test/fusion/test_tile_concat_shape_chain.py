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
"""Tile repeats computed from a shape-only chain that stays outside the fusion.

A fused node cannot declare a CPU input, so ORT copies a runtime repeats tensor
onto the device and the kernel would read it back with a device-to-host copy and
a compute-stream synchronize. The Tile+Concat matcher therefore proves that
repeats is a pure function of the shapes of the fused inputs and hands that
expression to the fused node, which recomputes the value instead of reading it.

The chain itself is NOT absorbed: the dnn models read its intermediate values
from other branches, and a fused node owns every value it absorbs. These tests
pin that behaviour down:

* a chain with Gather + Reshape whose intermediate value is also a graph output
  still fuses (previously both conditions vetoed the fusion),
* repeats follows another tensor's shape while the Tile input keeps its shape,
* a chain that is not shape-only, and a chain whose Shape source is not an input
  of Tile+Concat, leave the Tile standalone and still produce the right result.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from op_test_utils import musa_devices

DUMP_ENV = "ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID"
DUMP_PATH_ENV = "ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID_PATH"
REPEATS_DIAGNOSTIC = "MUSA_TILE_CONCAT_REPEATS"
FALLBACK_DIAGNOSTIC = "shape_expr=0"
RECOMPUTED_DIAGNOSTIC = "shape_expr=1"
OPS_DIR = Path(__file__).resolve().parents[1] / "ops"

_SUBPROCESS_RUNNER = """
import os
import sys

import numpy as np
import onnxruntime as ort

from op_test_utils import musa_devices

model_path, feeds_path = sys.argv[1], sys.argv[2]
with open(model_path, "rb") as handle:
    model = handle.read()
feeds = dict(np.load(feeds_path))

options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
options.add_provider_for_devices(musa_devices(), {})
session = ort.InferenceSession(model, sess_options=options)
session.run(None, feeds)
"""


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


def _int64_scalar(name, value):
    return helper.make_tensor(name, TensorProto.INT64, [], [value])


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
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
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


def _run_in_subprocess(model, feeds, tmp_path):
    """Run once in a fresh process and return (runtime dump, stderr).

    The runtime graph dump is written at process exit, and the diagnostics the
    Tile+Concat factory prints are only emitted while the session is created, so
    both have to be collected outside the pytest process.
    """
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(model)
    feeds_path = tmp_path / "feeds.npz"
    np.savez(feeds_path, **feeds)
    dump_path = tmp_path / "runtime_graph.mmd"
    env = dict(os.environ)
    env[DUMP_ENV] = "1"
    env[DUMP_PATH_ENV] = str(dump_path)
    # The subprocess imports op_test_utils the way conftest.py makes it visible.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(OPS_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _SUBPROCESS_RUNNER,
            str(model_path),
            str(feeds_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    dump = dump_path.read_text(encoding="utf-8") if dump_path.exists() else ""
    return dump, completed.stderr


def _batch_chain_nodes(repeats_source):
    return [
        helper.make_node("Shape", ["query"], ["query_shape"]),
        helper.make_node("Gather", ["query_shape", "zero_scalar"], ["batch"], axis=0),
        *repeats_source,
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y_pre"], axis=1),
        helper.make_node("Relu", ["y_pre"], ["y"]),
    ]


def _batch_chain_initializers():
    return [
        _int64_initializer("zero", [0]),
        _int64_scalar("zero_scalar", 0),
        _int64_initializer("one", [1]),
        _float_initializer("data", np.arange(4, dtype=np.float32), (1, 4)),
    ]


def test_tile_concat_repeats_follows_another_tensors_shape(tmp_path):
    """One session, two batch sizes: repeats moves while the Tile input does not.

    `data` stays [1, 4] and only `query`'s batch axis changes, so a constant
    repeats would be wrong for one of the two runs.
    """
    rng = np.random.default_rng(29)
    nodes = _batch_chain_nodes(
        [
            helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
            helper.make_node("Concat", ["batch_1d", "one"], ["repeats"], axis=0),
        ]
    )
    model = _build_model(
        "tile_concat_repeats_follows_shape",
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

    ops = _read_ops_by_provider(musa)
    assert _fused_ops(ops)
    assert "Tile" not in _musa_ops(ops)

    dump, stderr = _run_in_subprocess(model, {"query": query}, tmp_path)
    assert "TileConcatFusionCompute" in dump
    assert REPEATS_DIAGNOSTIC in stderr
    assert RECOMPUTED_DIAGNOSTIC in stderr
    assert FALLBACK_DIAGNOSTIC not in stderr


def test_tile_concat_repeats_chain_with_gather_reshape_and_external_consumer(
    tmp_path,
):
    """A Gather+Reshape chain whose intermediate value is a graph output fuses.

    Both properties used to veto the fusion: a chain value with an outside
    consumer, and a body that carries Gather + Concat + Reshape (which the loose
    IsShapeReshapeFusionGraph detector could claim). The chain now stays in the
    graph, the fused body is only Tile + Concat, and the fused node is still the
    TileConcat compute.
    """
    rng = np.random.default_rng(31)
    nodes = [
        helper.make_node("Shape", ["query"], ["query_shape"]),
        helper.make_node("Gather", ["query_shape", "zero_scalar"], ["batch"], axis=0),
        helper.make_node("Reshape", ["batch", "one_shape"], ["batch_1d"]),
        helper.make_node("Concat", ["batch_1d", "one"], ["repeats"], axis=0),
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y_pre"], axis=1),
        helper.make_node("Relu", ["y_pre"], ["y"]),
    ]
    initializers = _batch_chain_initializers() + [_int64_initializer("one_shape", [1])]
    model = _build_model(
        "tile_concat_repeats_gather_reshape_external_consumer",
        nodes,
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, ["batch", 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("batch", TensorProto.INT64, None),
        ],
        initializers,
    )

    cpu = _session(model, use_musa=False)
    musa = _session(
        model, use_musa=True, profile_prefix=str(tmp_path / "gather_reshape")
    )
    for batch in (2, 5):
        query = rng.standard_normal((batch, 4)).astype(np.float32)
        feeds = {"query": query}
        expected = cpu.run(None, feeds)
        actual = musa.run(None, feeds)
        assert len(actual) == 2
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[1] == np.int64(batch)

    ops = _read_ops_by_provider(musa)
    assert _fused_ops(ops)
    assert "Tile" not in _musa_ops(ops)

    dump, stderr = _run_in_subprocess(model, {"query": query}, tmp_path)
    # The factory ORT picked for this fused node, not its detector order.
    assert "TileConcatFusionCompute" in dump
    assert "ShapeReshapeFusionCompute" not in dump
    assert RECOMPUTED_DIAGNOSTIC in stderr
    assert FALLBACK_DIAGNOSTIC not in stderr


def test_tile_concat_repeats_chain_with_non_shape_op_stays_standalone(tmp_path):
    """A chain with an op outside the shape-only set keeps the standalone Tile.

    `Max` is not shape arithmetic, so the matcher proves nothing. The Tile runs
    on its own (reading repeats, as before), the Concat is not fused, and both
    outputs still match the CPU.
    """
    rng = np.random.default_rng(37)
    nodes = [
        helper.make_node("Shape", ["query"], ["query_shape"]),
        helper.make_node("Gather", ["query_shape", "zero_scalar"], ["batch"], axis=0),
        helper.make_node("Max", ["batch", "one"], ["max_batch"]),
        helper.make_node("Reshape", ["max_batch", "one_shape"], ["batch_1d"]),
        helper.make_node("Concat", ["batch_1d", "one"], ["repeats"], axis=0),
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y_pre"], axis=1),
        helper.make_node("Relu", ["y_pre"], ["y"]),
    ]
    initializers = _batch_chain_initializers() + [_int64_initializer("one_shape", [1])]
    model = _build_model(
        "tile_concat_repeats_non_shape_op",
        nodes,
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )

    query = rng.standard_normal((3, 4)).astype(np.float32)
    feeds = {"query": query}
    expected = _session(model, use_musa=False).run(None, feeds)[0]
    musa = _session(
        model, use_musa=True, profile_prefix=str(tmp_path / "non_shape_op")
    )
    actual = musa.run(None, feeds)[0]
    assert actual.shape == expected.shape == (3, 8)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    musa_ops = _musa_ops(_read_ops_by_provider(musa))
    assert "Tile" in musa_ops
    assert "Concat" in musa_ops


def test_tile_concat_repeats_source_outside_fusion_stays_standalone(tmp_path):
    """A Shape source that Tile+Concat does not read cannot be recomputed.

    `other` only feeds the chain, so it is not an input of the fused node and
    the expression would need a tensor the fusion never sees. The Tile keeps its
    standalone kernel, which reads repeats, and the result stays correct.
    """
    rng = np.random.default_rng(41)
    nodes = [
        helper.make_node("Shape", ["other"], ["other_shape"]),
        helper.make_node("Gather", ["other_shape", "zero_scalar"], ["other_dim"], axis=0),
        helper.make_node("Unsqueeze", ["other_dim", "zero"], ["other_dim_1d"]),
        helper.make_node("Concat", ["other_dim_1d", "one"], ["repeats"], axis=0),
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y_pre"], axis=1),
        helper.make_node("Relu", ["y_pre"], ["y"]),
    ]
    initializers = _batch_chain_initializers()
    model = _build_model(
        "tile_concat_repeats_source_outside_fusion",
        nodes,
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("other", TensorProto.FLOAT, ["m"]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )

    feeds = {
        "query": rng.standard_normal((3, 4)).astype(np.float32),
        "other": rng.standard_normal(3).astype(np.float32),
    }
    expected = _session(model, use_musa=False).run(None, feeds)[0]
    musa = _session(
        model, use_musa=True, profile_prefix=str(tmp_path / "source_outside")
    )
    actual = musa.run(None, feeds)[0]
    assert actual.shape == expected.shape == (3, 8)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    musa_ops = _musa_ops(_read_ops_by_provider(musa))
    assert "Tile" in musa_ops
    assert "Concat" in musa_ops


def test_tile_concat_repeats_chain_with_negative_slice():
    """Negative Slice bounds are resolved against the runtime shape."""
    rng = np.random.default_rng(43)
    nodes = [
        helper.make_node("Shape", ["query"], ["query_shape"]),
        helper.make_node(
            "Slice", ["query_shape", "head_start", "head_end"], ["batch_dim"]
        ),
        helper.make_node("Concat", ["batch_dim", "one"], ["repeats"], axis=0),
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y_pre"], axis=1),
        helper.make_node("Relu", ["y_pre"], ["y"]),
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
        helper.make_node("Concat", ["tiled", "query"], ["y_pre"], axis=1),
        helper.make_node("Relu", ["y_pre"], ["y"]),
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


def test_tile_concat_runtime_repeats_switch_fuses_outside_source(
    tmp_path, monkeypatch
):
    """With runtime repeats enabled the outside-source Tile joins the fusion.

    The fused kernel then reads the repeats tensor once per execution instead
    of leaving the Tile standalone; the result must stay identical.
    """
    monkeypatch.setenv("ORT_MUSA_TILE_CONCAT_RUNTIME_REPEATS", "1")
    rng = np.random.default_rng(42)
    nodes = [
        helper.make_node("Shape", ["other"], ["other_shape"]),
        helper.make_node("Gather", ["other_shape", "zero_scalar"], ["other_dim"], axis=0),
        helper.make_node("Unsqueeze", ["other_dim", "zero"], ["other_dim_1d"]),
        helper.make_node("Concat", ["other_dim_1d", "one"], ["repeats"], axis=0),
        helper.make_node("Tile", ["data", "repeats"], ["tiled"]),
        helper.make_node("Concat", ["tiled", "query"], ["y_pre"], axis=1),
        helper.make_node("Relu", ["y_pre"], ["y"]),
    ]
    initializers = _batch_chain_initializers()
    model = _build_model(
        "tile_concat_runtime_repeats_outside_source",
        nodes,
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("other", TensorProto.FLOAT, ["m"]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )
    feeds = {
        "query": rng.standard_normal((3, 4)).astype(np.float32),
        "other": rng.standard_normal(3).astype(np.float32),
    }
    expected = _session(model, use_musa=False).run(None, feeds)[0]
    musa = _session(
        model, use_musa=True, profile_prefix=str(tmp_path / "runtime_repeats")
    )
    actual = musa.run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    musa_ops = _musa_ops(_read_ops_by_provider(musa))
    assert any(str(op).startswith("MUSAExecutionProvider_") for op in musa_ops)
    assert "Tile" not in musa_ops
    assert "Concat" not in musa_ops
