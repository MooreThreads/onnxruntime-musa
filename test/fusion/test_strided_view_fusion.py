# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""End-to-end coverage for the dynamic Slice -> Concat StridedView fusion."""

import numpy as np
import pytest
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run_model_and_compare


def _scalar_int64(name, value):
    return helper.make_tensor(name, TensorProto.INT64, [1], [value])


@pytest.mark.parametrize("segments", [3, 8])
def test_strided_view_fuses_dynamic_sequence_reorder(segments):
    """[N*S, B, D] -> [S, B, N*D] for any N > 1."""
    rng = np.random.default_rng(17)
    a = rng.standard_normal((segments * 5, 2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 7)).astype(np.float32)
    nodes = [helper.make_node("MatMul", ["A", "B"], ["X"]),
             helper.make_node("Shape", ["X"], ["shape"]),
             helper.make_node("Gather", ["shape", "axis_one"], ["length"], axis=0),
             helper.make_node("Add", ["length", "round_bias"], ["rounded"]),
             helper.make_node("Div", ["rounded", "segments"], ["block"])]
    initializers = [_scalar_int64("axis_one", 1),
                    _scalar_int64("round_bias", segments - 1),
                    _scalar_int64("segments", segments),
                    _scalar_int64("zero", 0),
                    _scalar_int64("slice_axis", 0)]
    bounds = ["zero"]
    for i in range(1, segments + 1):
        factor = f"factor_{i}"
        bound = f"bound_{i}"
        initializers.append(_scalar_int64(factor, i))
        nodes.append(helper.make_node("Mul", ["block", factor], [bound]))
        bounds.append(bound)
    slice_outputs = []
    for i in range(segments):
        output = f"slice_{i}"
        nodes.append(helper.make_node("Slice", ["X", bounds[i], bounds[i + 1], "slice_axis"], [output]))
        slice_outputs.append(output)
    nodes.append(helper.make_node("Concat", slice_outputs, ["Y"], axis=2))
    model = build_graph_model(
        nodes,
        {"A": a, "B": b},
        [("Y", TensorProto.FLOAT)],
        initializers=initializers,
        name="strided_view_dynamic_sequence_reorder",
    )
    run_model_and_compare(model, {"A": a, "B": b}, rtol=1e-5, atol=1e-5)
