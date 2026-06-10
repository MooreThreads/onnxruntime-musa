# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end test for the Split -> Reduce fusion pattern."""

import numpy as np
from onnx import helper

from op_test_utils import TensorProto, build_graph_model, run_model_and_compare


def test_split_reduce_prod_mean_fusion():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.5, 1.5, size=(4, 5, 3)).astype(np.float32)
    split = helper.make_tensor("split", TensorProto.INT64, [2], [2, 3])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Split", ["X", "split"], ["S0", "S1"], axis=1),
        helper.make_node("ReduceProd", ["S0", "axes"], ["Y0"], keepdims=0),
        helper.make_node("ReduceMean", ["S1", "axes"], ["Y1"], keepdims=0),
    ]
    feeds = {"X": x}
    model = build_graph_model(
        nodes,
        feeds,
        [("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        initializers=[split, axes],
        opset=18,
        name="split_reduce_fusion_graph",
    )

    run_model_and_compare(model, feeds, rtol=1e-4, atol=1e-4)
