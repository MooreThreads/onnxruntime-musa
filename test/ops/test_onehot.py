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
"""End-to-end CPU-vs-MUSA tests for the OneHot operator."""

import numpy as np
from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    run_and_compare,
    run_model_and_compare,
)


def test_onehot_jd_model_shape_int64_indices_int32_depth_float_values():
    indices = np.array([[0], [3], [6]], dtype=np.int64)
    depth = np.array([7], dtype=np.int32)
    values = np.array([0.0, 1.0], dtype=np.float32)
    run_and_compare(
        "OneHot",
        inputs={"indices": indices, "depth": depth, "values": values},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
        opset=18,
    )


def test_onehot_negative_and_out_of_range_indices():
    indices = np.array([[0, -1, 4, -5]], dtype=np.int64)
    depth = np.array([4], dtype=np.int32)
    values = np.array([-2.0, 3.0], dtype=np.float32)
    run_and_compare(
        "OneHot",
        inputs={"indices": indices, "depth": depth, "values": values},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": -1},
        opset=18,
    )


def test_onehot_tile_mul_chain():
    indices = np.array([[0], [2]], dtype=np.int64)
    depth = np.array([3], dtype=np.int32)
    values = np.array([0.0, 1.0], dtype=np.float32)
    repeats = np.array([1, 2, 1], dtype=np.int64)
    scale = np.full((2, 6, 1), 2.0, dtype=np.float32)

    nodes = [
        helper.make_node(
            "OneHot",
            ["indices", "depth", "values"],
            ["onehot"],
            axis=1,
        ),
        helper.make_node("Tile", ["onehot", "repeats"], ["tiled"]),
        helper.make_node("Mul", ["tiled", "scale"], ["Y"]),
    ]
    model = build_graph_model(
        nodes,
        inputs={
            "indices": indices,
            "depth": depth,
            "values": values,
            "repeats": repeats,
            "scale": scale,
        },
        outputs=[("Y", TensorProto.FLOAT)],
        opset=18,
        name="onehot_tile_mul",
    )
    run_model_and_compare(
        model,
        {
            "indices": indices,
            "depth": depth,
            "values": values,
            "repeats": repeats,
            "scale": scale,
        },
    )
