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
"""Operators whose nodes reach the EP only in particular opsets, dtypes or
graph shapes - the residue left after the opset and fusion work of the previous
steps. Each family below explains itself, and each of its cases builds the graph
from the model zoo, runs it on the stock CPU EP and on the MUSA EP, and checks
both the numbers and the placement: the MUSA session disables the CPU EP
fallback and asks the EP's dtype diagnostics to report the assignment of every
node, so an operator the EP does not claim fails loudly instead of silently
running on the CPU EP.
"""

from __future__ import annotations

import collections
import re

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    build_model,
    musa_devices,
    run,
)

Case = collections.namedtuple(
    "Case", "op_type opset inputs outputs attrs label", defaults=(None, None)
)
GeluSubgraphCase = collections.namedtuple(
    "GeluSubgraphCase", "opset with_half model_bytes feeds output_name"
)
LayerNormCase = collections.namedtuple(
    "LayerNormCase", "opset dtype_name rtol model_bytes feeds"
)
UnaryCase = collections.namedtuple(
    "UnaryCase", "op_type opset dtype_name rtol model_bytes feeds"
)
AveragePoolCase = collections.namedtuple(
    "AveragePoolCase", "label opset x tensor_type rtol attrs"
)
GeluCase = collections.namedtuple(
    "GeluCase", "opset dtype_name rtol model_bytes feeds"
)

# The opsets the zoo models use for the old range, and the current one.
_LOW_OPSETS = (10, 12)


def _options(devices, *, disable_fallback=True):
    options = ort.SessionOptions()
    if disable_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {"dtype_diagnostics": "1"})
    return options


def _report(capfd):
    return [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("MUSA_DTYPE_NODE")
    ]


def _assert_all_claimed(capfd, op_types, context):
    """Every reported node is one of op_types and a MUSA kernel took it."""
    lines = _report(capfd)
    assert len(lines) == len(op_types), f"{context}: {lines}"
    seen = sorted(re.search(r" op=(\S+) ", line).group(1) for line in lines)
    assert seen == sorted(op_types), f"{context}: {seen}"
    for line in lines:
        assert line.rstrip().endswith("assigned=1"), f"{context}: {line}"


def _require_devices():
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for the fusion ops test")
    return devices


def _assert_claimed_and_equal(capfd, model_bytes, feeds, opset, rtol, atol,
                              op_type, expected_ops=None):
    """Run the graph on the CPU EP and on the MUSA EP, compare, and assert that
    every reported node was claimed by a MUSA kernel."""
    context = f"{op_type} at opset {opset}"
    devices = _require_devices()
    cpu_outputs = run(model_bytes, feeds, use_musa=False)

    capfd.readouterr()
    session = ort.InferenceSession(model_bytes, sess_options=_options(devices))
    musa_outputs = session.run(None, dict(feeds))

    assert len(cpu_outputs) == len(musa_outputs), f"{context}: output count"
    for index, (cpu_output, musa_output) in enumerate(zip(cpu_outputs, musa_outputs)):
        assert cpu_output.shape == musa_output.shape, f"{context}: output[{index}] shape"
        assert cpu_output.dtype == musa_output.dtype, f"{context}: output[{index}] dtype"
        np.testing.assert_allclose(
            musa_output.astype(np.float64),
            cpu_output.astype(np.float64),
            rtol=rtol,
            atol=atol,
            err_msg=f"{context}: output[{index}] CPU vs MUSA mismatch",
        )
    _assert_all_claimed(capfd, expected_ops or [op_type], context)
    return musa_outputs


# --------------------------------------------------------------------------- #
# Erf, and the Gelu subgraph whose fusion the EP's partitioning interrupts
# --------------------------------------------------------------------------- #
#
# ORT fuses the Bert/GPT activation ``x * 0.5 * (1 + erf(x / sqrt(2)))`` into a
# single Gelu node in a level-2 pass that runs *after* the EP answered
# GetCapability - the fused node is built from nodes the EP had already claimed,
# and the EP is never asked about it. While the whole subgraph sat on the CPU EP
# that fusion matched; once the surrounding Div/Add/Mul move to the MUSA EP it
# no longer does, and the raw Erf node is left behind - registered from opset 13
# on only, so an opset 10-12 model kept one Erf per Bert layer on the CPU EP.
# The subgraph case below is the evidence for both halves: the chain keeps all
# of its nodes on MUSA, and the node count proves ORT did not fuse it into a
# Gelu the EP was never asked about.


def _erf_cases():
    rng = np.random.default_rng(41)
    x = rng.uniform(-3.0, 3.0, (2, 5, 4)).astype(np.float32)
    for opset in (9, 10, 11, 12):
        yield Case("Erf", opset, {"X": x}, [("Y", TensorProto.FLOAT)])
    yield Case("Erf", 12, {"X": x.astype(np.float16)}, [("Y", TensorProto.FLOAT16)],
               None, "float16")


def _gelu_graph_models():
    """The two shapes of the Bert/Gelu pattern ORT fuses into one Gelu node.

    ``div -> Erf -> Add(1) -> Mul`` with the root multiplied in, plus the
    variant that scales the whole result by 0.5.
    """
    rng = np.random.default_rng(43)
    x = rng.uniform(-3.0, 3.0, (1, 4, 8)).astype(np.float32)
    sqrt_two = np.array([1.4142099618911743], dtype=np.float32)
    one = np.array([1.0], dtype=np.float32)
    half = np.array([0.5], dtype=np.float32)
    for opset in (11, 12):
        for with_half in (False, True):
            nodes = [
                helper.make_node("Div", ["X", "sqrt_two"], ["div"]),
                helper.make_node("Erf", ["div"], ["erf"]),
                helper.make_node("Add", ["erf", "one"], ["add"]),
                helper.make_node("Mul", ["X", "add"], ["mul"]),
            ]
            outputs = [("mul", TensorProto.FLOAT)]
            if with_half:
                nodes.append(helper.make_node("Mul", ["mul", "half"], ["y"]))
                outputs = [("y", TensorProto.FLOAT)]
            model_bytes = build_graph_model(
                nodes,
                {"X": x},
                outputs,
                initializers=[
                    helper.make_tensor("sqrt_two", TensorProto.FLOAT, [1], sqrt_two),
                    helper.make_tensor("one", TensorProto.FLOAT, [1], one),
                    helper.make_tensor("half", TensorProto.FLOAT, [1], half),
                ],
                opset=opset,
                name=f"gelu_pattern_{opset}_{with_half}",
            )
            yield GeluSubgraphCase(
                opset, with_half, model_bytes, {"X": x}, outputs[-1][0]
            )


@pytest.mark.parametrize("case", list(_erf_cases()), ids=lambda case: f"Erf-opset{case.opset}-{case.label or 'float32'}")
def test_erf_old_opsets(capfd, case):
    model_bytes = build_model(
        case.op_type, case.inputs, case.outputs, case.attrs, "", case.opset
    )
    rtol, atol = (2e-2, 2e-2) if case.label == "float16" else (1e-5, 1e-6)
    _assert_claimed_and_equal(
        capfd, model_bytes, case.inputs, case.opset, rtol, atol, case.op_type,
    )


@pytest.mark.parametrize(
    "case",
    list(_gelu_graph_models()),
    ids=lambda case: f"opset{case.opset}-{'half' if case.with_half else 'no-half'}",
)
def test_gelu_subgraph_is_fully_on_musa(capfd, case):
    opset, with_half = case.opset, case.with_half
    model_bytes, feeds = case.model_bytes, case.feeds
    """The Div/Erf/Add/Mul chain keeps all of its nodes on the MUSA EP."""
    devices = _require_devices()
    cpu_outputs = run(model_bytes, feeds, use_musa=False)

    capfd.readouterr()
    session = ort.InferenceSession(model_bytes, sess_options=_options(devices))
    musa_outputs = session.run(None, dict(feeds))
    np.testing.assert_allclose(
        musa_outputs[0], cpu_outputs[0], rtol=1e-5, atol=1e-6
    )
    # 4 nodes for the plain pattern, 5 with the trailing Mul(0.5). A different
    # count would mean ORT fused the subgraph into a single Gelu node, which the
    # EP is never asked about.
    expected = ["Div", "Erf", "Add", "Mul"] + (["Mul"] if with_half else [])
    _assert_all_claimed(capfd, expected, f"gelu pattern at opset {opset}")


# --------------------------------------------------------------------------- #
# float16 LayerNormalization
# --------------------------------------------------------------------------- #
#
# ORT binds LayerNormalization's Scale/Bias to the same type parameter as X
# (the ONNX schema calls it U and only allows float/bfloat16 there), so the
# float16 layers the siglip and bge_reranker models carry bring float16 Scale
# and Bias with them. The kernel reads Scale/Bias through the input element
# type, and only the float32 route goes through muDNN, which wants float32
# Scale/Bias.


def _layer_norm_models():
    rng = np.random.default_rng(47)
    for opset in (17, 19):
        for dtype, tensor_type, rtol in (
            (np.float16, TensorProto.FLOAT16, 2e-2),
            (np.float32, TensorProto.FLOAT, 1e-4),
        ):
            x = rng.standard_normal((2, 3, 8)).astype(dtype)
            scale = np.linspace(0.5, 1.5, 8).astype(dtype)
            bias = np.linspace(-0.2, 0.2, 8).astype(dtype)
            model_bytes = build_model(
                "LayerNormalization",
                {"X": x, "Scale": scale, "B": bias},
                [("Y", tensor_type)],
                {"axis": -1, "epsilon": 1e-5},
                "",
                opset,
            )
            yield LayerNormCase(
                opset,
                np.dtype(dtype).name,
                rtol,
                model_bytes,
                {"X": x, "Scale": scale, "B": bias},
            )


@pytest.mark.parametrize(
    "case",
    list(_layer_norm_models()),
    ids=lambda case: f"opset{case.opset}-{case.dtype_name}",
)
def test_layer_normalization_dtype_matrix(capfd, case):
    _assert_claimed_and_equal(
        capfd,
        case.model_bytes,
        case.feeds,
        case.opset,
        case.rtol,
        case.rtol,
        "LayerNormalization",
    )


def test_layer_normalization_float16_fused_into_the_graph(capfd):
    """The dtype combination the siglip/bge_reranker graphs carry."""
    devices = _require_devices()
    rng = np.random.default_rng(53)
    x = rng.standard_normal((1, 6, 16)).astype(np.float16)
    scale = np.linspace(0.5, 1.5, 16).astype(np.float16)
    bias = np.linspace(-0.2, 0.2, 16).astype(np.float16)
    model_bytes = build_graph_model(
        [
            helper.make_node(
                "LayerNormalization", ["X", "Scale", "B"], ["norm"]
            ),
            helper.make_node("Relu", ["norm"], ["Y"]),
        ],
        {"X": x},
        [("Y", TensorProto.FLOAT16)],
        initializers=[
            helper.make_tensor("Scale", TensorProto.FLOAT16, [16], scale),
            helper.make_tensor("B", TensorProto.FLOAT16, [16], bias),
        ],
        opset=17,
        name="layer_norm_float16_graph",
    )
    feeds = {"X": x}
    cpu_outputs = run(model_bytes, feeds, use_musa=False)
    capfd.readouterr()
    session = ort.InferenceSession(model_bytes, sess_options=_options(devices))
    musa_outputs = session.run(None, dict(feeds))
    np.testing.assert_allclose(
        musa_outputs[0].astype(np.float64),
        cpu_outputs[0].astype(np.float64),
        rtol=2e-2,
        atol=2e-2,
    )
    _assert_all_claimed(capfd, ["LayerNormalization", "Relu"], "float16 LayerNorm graph")


# --------------------------------------------------------------------------- #
# Sin / Cos
# --------------------------------------------------------------------------- #
#
# The Sin/Cos schemas are unchanged from opset 7 through 19 apart from the
# element types added at 13, so one registration covers the positional-encoding
# nodes of the opset 13-21 models and the older ones alike.


def _sin_cos_models():
    rng = np.random.default_rng(59)
    for op_type in ("Sin", "Cos"):
        for opset in (7, 12, 19):
            for dtype, tensor_type, rtol in (
                (np.float32, TensorProto.FLOAT, 1e-5),
                (np.float16, TensorProto.FLOAT16, 2e-2),
            ):
                x = rng.uniform(-6.0, 6.0, (2, 3, 4)).astype(dtype)
                model_bytes = build_model(
                    op_type, {"X": x}, [("Y", tensor_type)], None, "", opset
                )
                yield UnaryCase(
                    op_type, opset, np.dtype(dtype).name, rtol, model_bytes, {"X": x}
                )


@pytest.mark.parametrize(
    "case",
    list(_sin_cos_models()),
    ids=lambda case: f"{case.op_type}-opset{case.opset}-{case.dtype_name}",
)
def test_sin_cos_matrix(capfd, case):
    _assert_claimed_and_equal(
        capfd,
        case.model_bytes,
        case.feeds,
        case.opset,
        case.rtol,
        case.rtol,
        case.op_type,
    )


# --------------------------------------------------------------------------- #
# AveragePool
# --------------------------------------------------------------------------- #
#
# AveragePool has no dilations but a count_include_pad switch the pooling route
# has to model, and a ceil_mode this route does not implement: the capability
# predicate and the kernel share one rejection list
# (kernels/nn/average_pool_support.h), so a node this route cannot run stays on
# the CPU EP instead of being claimed and failing when it runs. The last case
# below is that negative half.


def _average_pool_models():
    rng = np.random.default_rng(61)
    x4 = rng.uniform(-2.0, 2.0, (1, 3, 8, 8)).astype(np.float32)
    x5 = rng.uniform(-2.0, 2.0, (1, 2, 6, 6, 6)).astype(np.float32)
    x16 = x4.astype(np.float16)
    # The opset-7 schema already has kernel_shape/strides/pads/count_include_pad.
    yield AveragePoolCase("opset7-k2s2", 7, x4, TensorProto.FLOAT, 1e-5,
                          {"kernel_shape": [2, 2], "strides": [2, 2]})
    yield AveragePoolCase("opset10-k2s2-ceil0", 10, x4, TensorProto.FLOAT, 1e-5,
                          {"kernel_shape": [2, 2], "strides": [2, 2], "ceil_mode": 0})
    yield AveragePoolCase(
        "opset11-pads-count", 11, x4, TensorProto.FLOAT, 1e-5,
        {"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1],
         "count_include_pad": 1})
    yield AveragePoolCase(
        "opset11-pads-no-count", 11, x4, TensorProto.FLOAT, 1e-5,
        {"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1],
         "count_include_pad": 0})
    yield AveragePoolCase(
        "opset11-asymmetric-pads", 11, x4, TensorProto.FLOAT, 1e-5,
        {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [0, 1, 1, 0]})
    yield AveragePoolCase(
        "opset19-same-upper", 19, x4, TensorProto.FLOAT, 1e-5,
        {"kernel_shape": [2, 2], "strides": [2, 2], "auto_pad": "SAME_UPPER"})
    yield AveragePoolCase(
        "opset19-same-lower", 19, x4, TensorProto.FLOAT, 1e-5,
        {"kernel_shape": [3, 3], "strides": [2, 2], "auto_pad": "SAME_LOWER"})
    yield AveragePoolCase(
        "opset19-valid", 19, x4, TensorProto.FLOAT, 1e-5,
        {"kernel_shape": [3, 3], "strides": [2, 2], "auto_pad": "VALID"})
    yield AveragePoolCase("opset19-rank5", 19, x5, TensorProto.FLOAT, 1e-5,
                          {"kernel_shape": [2, 2, 2], "strides": [2, 2, 2]})
    # float16 stays at opset 11: at opset 19 the ORT NchwcTransformer rewrites a
    # float16 AveragePool into com.ms.internal.nhwc.AveragePool(19), which the
    # stock CPU EP has no kernel for - an ORT-side artefact this test must not
    # depend on.
    yield AveragePoolCase(
        "opset11-float16", 11, x16, TensorProto.FLOAT16, 2e-2,
        {"kernel_shape": [2, 2], "strides": [2, 2], "count_include_pad": 1})


@pytest.mark.parametrize(
    "case", list(_average_pool_models()), ids=lambda case: case.label
)
def test_average_pool_matrix(capfd, case):
    model_bytes = build_model(
        "AveragePool",
        {"X": case.x},
        [("Y", case.tensor_type)],
        case.attrs,
        "",
        case.opset,
    )
    _assert_claimed_and_equal(
        capfd, model_bytes, {"X": case.x}, case.opset, case.rtol, case.rtol,
        "AveragePool",
    )


@pytest.mark.parametrize("opset", [10, 19])
def test_average_pool_ceil_mode_stays_on_cpu(capfd, opset):
    """ceil_mode is outside this MUSA route, so the node must stay on the CPU EP
    instead of being claimed and failing when it runs."""
    devices = _require_devices()
    rng = np.random.default_rng(67)
    x = rng.uniform(-2.0, 2.0, (1, 2, 7, 7)).astype(np.float32)
    attrs = {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1}
    model_bytes = build_model(
        "AveragePool", {"X": x}, [("Y", TensorProto.FLOAT)], attrs, "", opset
    )
    cpu_outputs = run(model_bytes, feeds={"X": x}, use_musa=False)

    # The CPU EP fallback stays enabled here: the point of the case is that the
    # node is not claimed, and the numbers must still match.
    capfd.readouterr()
    session = ort.InferenceSession(
        model_bytes, sess_options=_options(devices, disable_fallback=False)
    )
    musa_outputs = session.run(None, {"X": x})
    np.testing.assert_allclose(musa_outputs[0], cpu_outputs[0], rtol=1e-5, atol=1e-6)
    lines = _report(capfd)
    assert len(lines) == 1 and lines[0].rstrip().endswith("assigned=0"), lines


# --------------------------------------------------------------------------- #
# The ONNX Gelu (opset 20)
# --------------------------------------------------------------------------- #
#
# Gelu became an official ONNX operator at opset 20 (and a level-1 fusion target
# for the models that export it), which is what the opset-21 models carry;
# the tanh variant is a different function and stays on the CPU EP.


def _gelu_op_models():
    rng = np.random.default_rng(71)
    for opset in (20, 21):
        for dtype, tensor_type, rtol in (
            (np.float32, TensorProto.FLOAT, 1e-5),
            (np.float16, TensorProto.FLOAT16, 2e-2),
        ):
            x = rng.uniform(-4.0, 4.0, (2, 3, 4)).astype(dtype)
            model_bytes = build_model(
                "Gelu", {"X": x}, [("Y", tensor_type)], None, "", opset
            )
            yield GeluCase(opset, np.dtype(dtype).name, rtol, model_bytes, {"X": x})


@pytest.mark.parametrize(
    "case",
    list(_gelu_op_models()),
    ids=lambda case: f"opset{case.opset}-{case.dtype_name}",
)
def test_gelu_opset20_matrix(capfd, case):
    _assert_claimed_and_equal(
        capfd, case.model_bytes, case.feeds, case.opset, case.rtol, case.rtol,
        "Gelu",
    )


def test_gelu_tanh_approximation_stays_on_cpu(capfd):
    """The tanh approximation is a different function; those nodes stay on the
    CPU EP rather than silently running the exact form."""
    devices = _require_devices()
    rng = np.random.default_rng(73)
    x = rng.uniform(-4.0, 4.0, (2, 3, 4)).astype(np.float32)
    model_bytes = build_model(
        "Gelu", {"X": x}, [("Y", TensorProto.FLOAT)], {"approximate": "tanh"}, "", 20
    )
    cpu_outputs = run(model_bytes, feeds={"X": x}, use_musa=False)
    capfd.readouterr()
    session = ort.InferenceSession(
        model_bytes, sess_options=_options(devices, disable_fallback=False)
    )
    musa_outputs = session.run(None, {"X": x})
    np.testing.assert_allclose(musa_outputs[0], cpu_outputs[0], rtol=1e-5, atol=1e-6)
    lines = _report(capfd)
    assert len(lines) == 1 and lines[0].rstrip().endswith("assigned=0"), lines


