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
"""Old-opset coverage for the kernels whose registration was extended down.

These kernels were previously only registered from opset 13 on, which left every
node of an opset 10/11/12 model on the CPU EP. The registrations now reuse the
same kernel class for the older opsets, so each case here builds the same
single-node graph at opset 10 and opset 12 - the opsets the low-opset models use
- and checks two things:

  1. the MUSA EP claims the node (the session runs with
     `session.disable_cpu_ep_fallback`, and the EP's dtype report must list the
     node as `assigned=1`), and
  2. its output matches the stock CPU EP.
"""

from __future__ import annotations

import collections

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from op_test_utils import TensorProto, build_model, musa_devices, run

Case = collections.namedtuple(
    "Case", "op_type opset inputs outputs attrs label", defaults=(None, None)
)

# `Shape` folds away when its input shape is static, so that case builds its
# model with a symbolic batch dimension instead.
_SYMBOLIC_BATCH_BUILDERS = {"Shape"}

# The lowest opsets the models in the zoo use; opset 11 sits between them and is
# covered by the same registration ranges.
_LOW_OPSETS = (10, 12)


def _build_model_with_symbolic_batch(op_type, inputs, outputs, attrs=None, opset=17):
    """Build a single-node graph whose first input dimension stays symbolic.

    `Shape` folds away at graph-optimization time when the input shape is fully
    static, and the graph would reach the EP empty. Keeping one dimension
    symbolic is also how the real models present it (a dynamic batch).
    """
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


def _binary_cases():
    rng = np.random.default_rng(7)
    a = rng.uniform(0.5, 2.0, (2, 1, 3)).astype(np.float32)
    b = rng.uniform(0.5, 2.0, (1, 4, 3)).astype(np.float32)
    for opset in _LOW_OPSETS:
        for op_type in ("Add", "Sub", "Mul", "Div", "Pow"):
            yield Case(op_type, opset, {"A": a, "B": b}, [("Y", TensorProto.FLOAT)])


def _unary_cases():
    rng = np.random.default_rng(11)
    x = rng.uniform(0.5, 2.0, (2, 3, 4)).astype(np.float32)
    for opset in _LOW_OPSETS:
        for op_type in (
            "Abs",
            "Exp",
            "Floor",
            "Log",
            "Neg",
            "Relu",
            "Sigmoid",
            "Sqrt",
        ):
            yield Case(op_type, opset, {"X": x}, [("Y", TensorProto.FLOAT)])


def _compare_cases():
    rng = np.random.default_rng(13)
    int_a = rng.integers(0, 3, (2, 3)).astype(np.int32)
    int_b = rng.integers(0, 3, (3,)).astype(np.int32)
    float_a = int_a.astype(np.float32)
    float_b = int_b.astype(np.float32)
    # Equal through opset 10 only accepts bool and integer tensors, and only
    # gains the floating-point types with the opset-11 schema.
    yield Case("Equal", 10, {"A": int_a, "B": int_b}, [("Y", TensorProto.BOOL)])
    yield Case("Equal", 11, {"A": float_a, "B": float_b}, [("Y", TensorProto.BOOL)])
    for op_type in ("Equal", "Less", "Greater"):
        yield Case(
            op_type,
            12,
            {"A": float_a, "B": float_b},
            [("Y", TensorProto.BOOL)],
        )
    for op_type in ("Less", "Greater"):
        yield Case(
            op_type,
            10,
            {"A": float_a, "B": float_b},
            [("Y", TensorProto.BOOL)],
        )


def _tensor_cases():
    rng = np.random.default_rng(17)
    x = rng.standard_normal((4, 6)).astype(np.float32)
    for opset in _LOW_OPSETS:
        yield Case(
            "Transpose",
            opset,
            {"X": x.reshape(2, 3, 4)},
            [("Y", TensorProto.FLOAT)],
            {"perm": [2, 0, 1]},
        )
        yield Case(
            "Reshape",
            opset,
            {"X": x, "shape": np.array([3, 8], dtype=np.int64)},
            [("Y", TensorProto.FLOAT)],
        )
        yield Case(
            "Gather",
            opset,
            {"data": x, "indices": np.array([0, 2, 5], dtype=np.int64)},
            [("Y", TensorProto.FLOAT)],
            {"axis": 1},
        )
        yield Case(
            "Concat",
            opset,
            {"A": x, "B": x[:, :2]},
            [("Y", TensorProto.FLOAT)],
            {"axis": 1},
        )
        yield Case(
            "Expand",
            opset,
            {"X": x[0], "shape": np.array([3, 6], dtype=np.int64)},
            [("Y", TensorProto.FLOAT)],
        )
        yield Case(
            "Cast",
            opset,
            {"X": x},
            [("Y", TensorProto.INT64)],
            {"to": TensorProto.INT64},
        )
        yield Case(
            "Shape",
            opset,
            {"X": x.reshape(2, 3, 4)},
            [("Y", TensorProto.INT64)],
            label="symbolic-batch",
        )
        yield Case(
            "NonZero",
            opset,
            {"X": np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32)},
            [("Y", TensorProto.INT64)],
        )
        yield Case(
            "MatMul",
            opset,
            {"A": x, "B": rng.standard_normal((6, 5)).astype(np.float32)},
            [("Y", TensorProto.FLOAT)],
        )


def _reduce_cases():
    rng = np.random.default_rng(19)
    x = rng.uniform(0.5, 2.0, (4, 8)).astype(np.float32)
    for opset in _LOW_OPSETS:
        for op_type in (
            "ReduceL2",
            "ReduceMax",
            "ReduceMean",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
        ):
            # opset 12 and older carry the axes as an attribute, not as an input.
            yield Case(
                op_type,
                opset,
                {"X": x},
                [("Y", TensorProto.FLOAT)],
                {"axes": [1], "keepdims": 1},
            )


def _reduce_sum_attribute_cases():
    rng = np.random.default_rng(31)
    x = rng.uniform(0.5, 2.0, (4, 8)).astype(np.float32)
    # ReduceSum is the one Reduce* whose axes moved from an attribute to an input
    # at opset 13 (the others follow at opset 18), so the opset-12-and-older form
    # is the attribute and the kernel has to read it instead of input 1.
    yield Case(
        "ReduceSum",
        11,
        {"X": x},
        [("Y", TensorProto.FLOAT)],
        {"axes": [1], "keepdims": 0},
        label="axes-keepdims0",
    )
    # The pre-opset-13 schemas make `axes` optional: omitting it reduces every
    # axis, which is what an empty axes attribute has to mean.
    yield Case(
        "ReduceSum",
        11,
        {"X": x},
        [("Y", TensorProto.FLOAT)],
        {"keepdims": 1},
        label="axes-omitted",
    )


def _conv_cases():
    x = np.random.default_rng(23).standard_normal((1, 2, 6, 6)).astype(np.float32)
    w = np.random.default_rng(29).standard_normal((3, 2, 3, 3)).astype(np.float32)
    # Conv-11 only clarifies the default strides and dilations, so opset 10 runs
    # the same kernel.
    yield Case(
        "Conv",
        10,
        {"X": x, "W": w},
        [("Y", TensorProto.FLOAT)],
        {"pads": [1, 1, 1, 1], "strides": [1, 1]},
    )


CASES = [
    *list(_binary_cases()),
    *list(_unary_cases()),
    *list(_compare_cases()),
    *list(_tensor_cases()),
    *list(_reduce_cases()),
    *list(_reduce_sum_attribute_cases()),
    *list(_conv_cases()),
]

CASE_IDS = [
    f"{case.op_type}-opset{case.opset}" + (f"-{case.label}" if case.label else "")
    for case in CASES
]

assert len(CASE_IDS) == len(set(CASE_IDS))
# Every operator whose registration was lowered has at least one case.
assert {case.op_type for case in CASES} == {
    "Abs", "Add", "Cast", "Concat", "Conv", "Div", "Equal", "Exp", "Expand",
    "Floor", "Gather", "Greater", "Less", "Log", "MatMul", "Mul", "Neg",
    "NonZero", "Pow",
    "ReduceL2", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceSum",
    "ReduceSumSquare", "Relu", "Reshape", "Shape", "Sigmoid", "Sqrt", "Sub",
    "Transpose",
}


def _run_case(capfd, case):
    """Run one case on CPU and MUSA; assert the CPU result and the assignment."""
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
        raise RuntimeError("No MUSA device available for the opset lower bound test")
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {"dtype_diagnostics": "1"})
    capfd.readouterr()  # drop whatever an earlier case printed
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

    # The dtype report is printed by GetCapability for every node of the graph;
    # this graph has one node, and `assigned=1` means a MUSA kernel took it.
    node_lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("MUSA_DTYPE_NODE")
    ]
    assert len(node_lines) == 1, f"{context}: expected one reported node, got {node_lines}"
    assert f" op={case.op_type} " in node_lines[0], f"{context}: {node_lines[0]}"
    assert node_lines[0].rstrip().endswith(
        "assigned=1"
    ), f"{context}: node not claimed by a MUSA kernel: {node_lines[0]}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_opset_lower_bound(capfd, case):
    _run_case(capfd, case)
