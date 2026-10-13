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
"""Old-opset semantics and the opset-20/21 upper bound.

Two families of changes are covered here, both of them invisible to a graph
that only uses the current opset:

* The operators whose meaning depends on the opset the model was stamped with.
  Softmax is the important one: through opset 12 the input is coerced into the
  2-D matrix [prod(d0..d_{axis-1}), prod(d_axis..)] and normalised along its
  second dimension (axis defaults to 1), while from opset 13 on only `axis` is
  reduced (default -1). Unsqueeze/Squeeze/Split/Clip/Pad/Slice/Gemm take their
  parameters from node attributes before the opset that turned them into
  inputs. Each case below builds the graph at the old opset, runs it on the CPU
  EP and on the MUSA EP, and checks both the numbers and that the MUSA EP
  actually claimed the node.

* The registrations whose upper bound was raised to meet models stamped with
  opset 20/21, where ONNX only added element types, so the opset-13 behaviour
  still covers the node.

The MUSA session runs with `session.disable_cpu_ep_fallback`, so an op, dtype or
opset the EP does not claim fails loudly instead of silently running on the CPU
EP; the `dtype_diagnostics` report is then checked for `assigned=1`.
"""

from __future__ import annotations

import collections

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, build_model, musa_devices, run

Case = collections.namedtuple(
    "Case", "op_type opset inputs outputs attrs label", defaults=(None, None)
)

# `Shape` folds away when its input shape is static, so that case keeps one
# dimension symbolic - the same builder the opset lower bound test uses.
_SYMBOLIC_BATCH_BUILDERS = {"Shape"}


def _build_model_with_symbolic_batch(op_type, inputs, outputs, attrs=None, opset=17):
    """Build a single-node graph whose first input dimension stays symbolic."""
    input_vis = []
    for name, array in inputs.items():
        shape = ["batch", *list(array.shape[1:])]
        input_vis.append(
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(array.dtype), shape
            )
        )
    output_vis = [helper.make_tensor_value_info(n, e, None) for n, e in outputs]
    node = helper.make_node(
        op_type, list(inputs.keys()), [n for n, _ in outputs], **(attrs or {})
    )
    graph = helper.make_graph([node], f"{op_type}_graph", input_vis, output_vis)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = min(model.ir_version, 10)
    return model.SerializeToString()


def _softmax_cases():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    tail_ones = rng.normal(size=(2, 3, 1, 1)).astype(np.float32)
    flat = rng.normal(size=(5,)).astype(np.float32)
    # Through opset 12 `axis` selects where the input is cut into the 2-D
    # matrix, so axis=1 normalises over the product of the trailing dimensions
    # while opset 13 normalises over dim 1 alone. Both are exercised, and the
    # last-axis case stays on the shared muDNN path in either semantics.
    yield Case("Softmax", 10, {"X": x}, [("Y", TensorProto.FLOAT)], {"axis": 1},
               "axis1-flattened")
    yield Case("Softmax", 11, {"X": x}, [("Y", TensorProto.FLOAT)], {"axis": 1},
               "axis1-flattened")
    yield Case("Softmax", 11, {"X": x}, [("Y", TensorProto.FLOAT)], {"axis": -1},
               "axis-1-last")
    yield Case("Softmax", 13, {"X": x}, [("Y", TensorProto.FLOAT)], {"axis": 1},
               "axis1-single")
    yield Case("Softmax", 11, {"X": tail_ones}, [("Y", TensorProto.FLOAT)], {"axis": 1},
               "axis1-tail-ones")
    # A 1-D input keeps its single axis: the flattened form then reduces over
    # the whole row of the promoted [1, d0] matrix.
    yield Case("Softmax", 11, {"X": flat}, [("Y", TensorProto.FLOAT)], {"axis": 0},
               "axis0-rank1")
    # The Softmax schema stops changing at opset 13, so an opset-21 model still
    # resolves the node to version 13 and the 13-17 registration matches it.
    yield Case("Softmax", 21, {"X": x}, [("Y", TensorProto.FLOAT)], {"axis": -1},
               "opset21")


def _squeeze_family_cases():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    squeeze_input = rng.normal(size=(1, 3, 1, 4)).astype(np.float32)
    split_input = rng.normal(size=(2, 5, 4)).astype(np.float32)
    even_split_input = rng.normal(size=(2, 6, 4)).astype(np.float32)
    # Unsqueeze/Squeeze/Split only move `axes`/`split` from attributes to inputs
    # at opset 13, and opset 11 documented negative axes.
    yield Case("Unsqueeze", 11, {"X": x}, [("Y", TensorProto.FLOAT)], {"axes": [0, 3]})
    yield Case("Unsqueeze", 11, {"X": x}, [("Y", TensorProto.FLOAT)],
               {"axes": [-1, -4]}, "negative-axes")
    yield Case("Unsqueeze", 21,
               {"X": x, "axes": np.array([0, 3], np.int64)},
               [("Y", TensorProto.FLOAT)], {}, "axes-input")
    yield Case("Squeeze", 11, {"X": squeeze_input}, [("Y", TensorProto.FLOAT)], {"axes": [0, 2]})
    yield Case("Squeeze", 11, {"X": squeeze_input}, [("Y", TensorProto.FLOAT)], {},
               "no-axes")
    yield Case("Squeeze", 21,
               {"X": squeeze_input, "axes": np.array([0, 2], np.int64)},
               [("Y", TensorProto.FLOAT)], {}, "axes-input")
    yield Case("Split", 11, {"X": split_input},
               [("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
               {"axis": 1, "split": [2, 3]})
    yield Case("Split", 10, {"X": even_split_input},
               [("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT), ("Y2", TensorProto.FLOAT)],
               {"axis": 1}, "equal-split")
    yield Case("Split", 11, {"X": even_split_input},
               [("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT), ("Y2", TensorProto.FLOAT)],
               {"axis": 1}, "equal-split")


def _parameter_attribute_cases():
    rng = np.random.default_rng(31)
    x = rng.uniform(-3.0, 3.0, (2, 3, 4)).astype(np.float32)
    xd = rng.uniform(-3.0, 3.0, (2, 4)).astype(np.float64)
    x4 = rng.uniform(-3.0, 3.0, (2, 2, 3, 4)).astype(np.float32)
    xi = rng.integers(-5, 6, (2, 7)).astype(np.int32)
    scaled = np.array(0.5, np.float32)
    # Clip kept min/max in FLOAT attributes through opset 10 and moved them to
    # inputs at opset 11 (which the CPU EP only implements for float32), with
    # the integer element types arriving at opset 12.
    yield Case("Clip", 10, {"X": x}, [("Y", TensorProto.FLOAT)],
               {"min": -1.0, "max": 1.5})
    yield Case("Clip", 10, {"X": x}, [("Y", TensorProto.FLOAT)], {"min": 0.0},
               "min-only")
    yield Case("Clip", 11, {"X": x, "min": scaled, "max": np.array(1.5, np.float32)},
               [("Y", TensorProto.FLOAT)])
    yield Case("Clip", 12, {"X": xi, "min": np.array(-2, np.int32),
                            "max": np.array(3, np.int32)},
               [("Y", TensorProto.INT32)], {}, "int32")
    # Pad kept pads and value in attributes through opset 10; `pads` carries one
    # begin/end pair per input axis.
    yield Case("Pad", 10, {"X": x4}, [("Y", TensorProto.FLOAT)],
               {"pads": [0, 1, 1, 0, 0, 0, 0, 1], "value": 2.5})
    yield Case("Pad", 10, {"X": xd}, [("Y", TensorProto.DOUBLE)],
               {"pads": [1, 0, 0, 1], "value": 0.25}, "double-value")
    yield Case("Pad", 11, {"X": x4, "pads": np.array([0, 1, 1, 0, 0, 0, 0, 1], np.int64),
                           "value": np.array(2.5, np.float32)},
               [("Y", TensorProto.FLOAT)], {}, "pads-input")
    # Slice kept starts/ends/axes in attributes through opset 9 (with the step
    # fixed to 1) and moved them to inputs at opset 10.
    yield Case("Slice", 9, {"X": x}, [("Y", TensorProto.FLOAT)],
               {"starts": [1], "ends": [3], "axes": [1]})
    yield Case("Slice", 9, {"X": x}, [("Y", TensorProto.FLOAT)],
               {"starts": [1, 1], "ends": [-1, 4]}, "implicit-axes")
    yield Case("Slice", 10, {"X": x, "starts": np.array([1], np.int64),
                             "ends": np.array([3], np.int64),
                             "axes": np.array([2], np.int64)},
               [("Y", TensorProto.FLOAT)], {}, "inputs")
    # Gemm's version changes are type sets and the optional C input.
    a = x[:2, :, 0]                  # [2, 3]
    b = x[0]                         # [3, 4]
    bias = x[1, 0]                   # [4]
    yield Case("Gemm", 7, {"A": a, "B": b, "C": bias}, [("Y", TensorProto.FLOAT)],
               {"alpha": 1.5, "beta": 0.5})
    yield Case("Gemm", 9, {"A": a.T.copy(), "B": b, "C": bias},
               [("Y", TensorProto.FLOAT)], {"transA": 1})
    yield Case("Gemm", 11, {"A": a, "B": b.T.copy()}, [("Y", TensorProto.FLOAT)],
               {"transB": 1}, "no-bias")
    yield Case("Gemm", 12, {"A": a, "B": b, "C": bias}, [("Y", TensorProto.FLOAT)], {})


def _upper_bound_cases():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    squeeze_input = rng.normal(size=(2, 3, 1)).astype(np.float32)
    reduce_input = rng.normal(size=(2, 3, 4)).astype(np.float32)
    # ONNX only added element types to these operators at opset 20/21, so
    # raising the registration's upper bound keeps the same behaviour.
    yield Case("Reshape", 21, {"X": x, "shape": np.array([3, 8], np.int64)},
               [("Y", TensorProto.FLOAT)], {}, "opset21")
    yield Case("Transpose", 21, {"X": x}, [("Y", TensorProto.FLOAT)], {"perm": [1, 0, 2]},
               "opset21")
    yield Case("Shape", 21, {"X": x}, [("Y", TensorProto.INT64)], {}, "opset21")
    yield Case("Unsqueeze", 21, {"X": x, "axes": np.array([0, 4], np.int64)},
               [("Y", TensorProto.FLOAT)], {}, "opset21")
    yield Case("Squeeze", 21, {"X": squeeze_input, "axes": np.array([2], np.int64)},
               [("Y", TensorProto.FLOAT)], {}, "opset21")
    yield Case("ReduceMax", 20, {"X": reduce_input, "axes": np.array([2], np.int64)},
               [("Y", TensorProto.FLOAT)], {"keepdims": 0}, "opset20")


CASES = [
    *list(_softmax_cases()),
    *list(_squeeze_family_cases()),
    *list(_parameter_attribute_cases()),
    *list(_upper_bound_cases()),
]

CASE_IDS = [
    f"{case.op_type}-opset{case.opset}" + (f"-{case.label}" if case.label else "")
    for case in CASES
]

assert len(CASE_IDS) == len(set(CASE_IDS))


def _assert_claimed(capfd, case, expected=True):
    """Check the assignment the EP reported for the graph's single node."""
    node_lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("MUSA_DTYPE_NODE")
    ]
    assert len(node_lines) == 1, f"{case.op_type} at opset {case.opset}: {node_lines}"
    assert f" op={case.op_type} " in node_lines[0], node_lines[0]
    expected_suffix = "assigned=1" if expected else "assigned=0"
    assert node_lines[0].rstrip().endswith(expected_suffix), node_lines[0]
    return node_lines[0]


def _run_case(capfd, case):
    """Run one case on the CPU EP and on the MUSA EP and compare."""
    context = f"{case.op_type} at opset {case.opset}"
    if case.op_type in _SYMBOLIC_BATCH_BUILDERS:
        model_bytes = _build_model_with_symbolic_batch(
            case.op_type, case.inputs, case.outputs, case.attrs, case.opset
        )
    else:
        model_bytes = build_model(
            case.op_type, case.inputs, case.outputs, case.attrs, "", case.opset
        )

    cpu_outputs = run(model_bytes, case.inputs, use_musa=False)

    devices = musa_devices()
    if not devices:
        raise RuntimeError(f"No MUSA device available for {context}")
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {"dtype_diagnostics": "1"})
    capfd.readouterr()
    session = ort.InferenceSession(model_bytes, sess_options=options)
    musa_outputs = session.run(None, dict(case.inputs))

    assert len(cpu_outputs) == len(musa_outputs), f"{context}: output count"
    for index, (cpu_output, musa_output) in enumerate(zip(cpu_outputs, musa_outputs)):
        assert cpu_output.shape == musa_output.shape, f"{context}: output[{index}] shape"
        assert cpu_output.dtype == musa_output.dtype, f"{context}: output[{index}] dtype"
        np.testing.assert_allclose(
            musa_output,
            cpu_output,
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"{context}: output[{index}] CPU vs MUSA mismatch",
        )

    _assert_claimed(capfd, case)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_opset_semantics(capfd, case):
    _run_case(capfd, case)


def _pad_float16_graphs():
    """The two Pad forms of one float16 padding, as (model, feeds) pairs.

    The opset-10 graph carries the pad value in the node's FLOAT `value`
    attribute, the opset-11 graph as a float16 input. Both pad a [1, 2] tensor
    by one element at the end of each axis.
    """
    rng = np.random.default_rng(41)
    x = rng.normal(size=(1, 2)).astype(np.float16)
    pads = np.array([0, 1, 0, 1], np.int64)
    # 0.1 is not representable in float16, so a truncating conversion instead
    # of the round-to-nearest one the ONNX reference uses would show up here.
    attribute_value = float(np.float32(0.1))
    attribute_model = build_model(
        "Pad",
        {"X": x},
        [("Y", TensorProto.FLOAT16)],
        {"pads": [0, 1, 0, 1], "value": attribute_value},
        "",
        10,
    )
    input_model = build_model(
        "Pad",
        {"X": x, "pads": pads, "value": np.array(np.float16(np.float32(0.1)))},
        [("Y", TensorProto.FLOAT16)],
        {},
        "",
        11,
    )
    # The float16 value stays a 0-d array: ORT rejects a bare numpy scalar feed.
    value_input = np.array(np.float16(np.float32(0.1)))
    return (
        (attribute_model, {"X": x}),
        (input_model, {"X": x, "pads": pads, "value": value_input}),
    )


def test_pad_float16_attribute_value_matches_the_reference(capfd):
    """The opset-10 Pad `value` attribute is a FLOAT that must be converted to
    the element type of the data.

    The CPU EP has no float16 Pad kernel at any opset, so the reference here is
    NumPy (and the opset-11 input form, which carries the float16 bits
    directly).
    """
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for the Pad float16 value test")

    attribute_graph, attribute_feeds = _pad_float16_graphs()[0]
    input_graph, input_feeds = _pad_float16_graphs()[1]
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {"dtype_diagnostics": "1"})

    capfd.readouterr()
    attribute_session = ort.InferenceSession(attribute_graph, sess_options=options)
    (attribute_output,) = attribute_session.run(None, dict(attribute_feeds))
    attribute_lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("MUSA_DTYPE_NODE")
    ]
    assert len(attribute_lines) == 1, attribute_lines
    assert attribute_lines[0].rstrip().endswith("assigned=1"), attribute_lines

    input_session = ort.InferenceSession(input_graph, sess_options=options)
    (input_output,) = input_session.run(None, dict(input_feeds))

    x = attribute_feeds["X"]
    expected = np.pad(x, [(0, 0), (1, 1)], constant_values=np.float16(np.float32(0.1)))
    np.testing.assert_array_equal(attribute_output, expected)
    np.testing.assert_array_equal(input_output, expected)
    np.testing.assert_array_equal(attribute_output, input_output)


def _slice_with_steps_graph(opset, step):
    """A Slice whose `steps` is a constant initializer, so the graph-level
    capability predicate can read it."""
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    negative = step < 0
    starts = [-1] if negative else [0]
    ends = [-2**63 + 1] if negative else [3]
    node = helper.make_node(
        "Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"]
    )
    initializers = [
        helper.make_tensor("starts", TensorProto.INT64, [1], starts),
        helper.make_tensor("ends", TensorProto.INT64, [1], ends),
        helper.make_tensor("axes", TensorProto.INT64, [1], [0]),
        helper.make_tensor("steps", TensorProto.INT64, [1], [step]),
    ]
    return (
        build_graph_model(
            [node], {"X": x}, [("Y", TensorProto.FLOAT)],
            initializers=initializers, opset=opset,
        ),
        {"X": x},
    )


def test_slice_step_sign_decides_the_claim(capfd):
    """A negative constant step cannot run on the MUSA device copy, so that node
    must be left to the CPU EP; a positive step is claimed and matches it."""
    devices = musa_devices()
    if not devices:
        raise RuntimeError("No MUSA device available for the Slice step test")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {"dtype_diagnostics": "1"})

    negative_graph, feeds = _slice_with_steps_graph(13, -1)
    capfd.readouterr()
    negative_session = ort.InferenceSession(negative_graph, sess_options=options)
    (negative_output,) = negative_session.run(None, dict(feeds))
    lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("MUSA_DTYPE_NODE")
    ]
    assert len(lines) == 1 and lines[0].rstrip().endswith("assigned=0"), lines
    np.testing.assert_array_equal(negative_output, feeds["X"][::-1])

    positive_graph, feeds = _slice_with_steps_graph(13, 2)
    cpu_outputs = run(positive_graph, feeds, use_musa=False)
    capfd.readouterr()
    positive_session = ort.InferenceSession(positive_graph, sess_options=options)
    (positive_output,) = positive_session.run(None, dict(feeds))
    lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("MUSA_DTYPE_NODE")
    ]
    assert len(lines) == 1 and lines[0].rstrip().endswith("assigned=1"), lines
    np.testing.assert_allclose(positive_output, cpu_outputs[0], rtol=1e-6, atol=1e-6)
