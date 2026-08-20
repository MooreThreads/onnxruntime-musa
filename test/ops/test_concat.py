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
"""End-to-end CPU-vs-MUSA test for the Concat operator."""

import numpy as np
import pytest
from onnx import helper

from op_test_utils import (
    TensorProto,
    build_graph_model,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_model_and_compare,
    run_with_iobinding,
)


def test_concat_axis0():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_concat_axis1():
    a = np.random.default_rng(2).standard_normal((32, 16)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_int64_negative_axis():
    a = np.arange(512, dtype=np.int64).reshape(16, 32)
    b = np.arange(512, 1024, dtype=np.int64).reshape(16, 32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": -1},
    )


def test_concat_int32_three_inputs_axis2():
    a = np.arange(2 * 3 * 1, dtype=np.int32).reshape(2, 3, 1)
    b = np.arange(100, 100 + 2 * 3 * 2, dtype=np.int32).reshape(2, 3, 2)
    c = np.arange(200, 200 + 2 * 3 * 3, dtype=np.int32).reshape(2, 3, 3)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axis": 2},
    )


def test_concat_bool_negative_axis():
    a = np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_)
    b = np.array([[[False, True], [True, False]], [[True, True], [False, False]]], dtype=np.bool_)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"axis": -1},
    )


def test_concat_float16_axis1():
    a = np.random.default_rng(4).standard_normal((4, 2)).astype(np.float16)
    b = np.random.default_rng(5).standard_normal((4, 3)).astype(np.float16)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axis": 1},
    )


def test_concat_uint8_axis0():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    b = np.arange(100, 108, dtype=np.uint8).reshape(2, 4)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.UINT8)],
        attrs={"axis": 0},
    )

@pytest.mark.parametrize(("dtype", "tensor_type"), [
    (np.uint16, TensorProto.UINT16), (np.uint32, TensorProto.UINT32),
    (np.uint64, TensorProto.UINT64), (np.int8, TensorProto.INT8),
    (np.int16, TensorProto.INT16), (np.float64, TensorProto.DOUBLE),
])
def test_concat_dtype_matrix(dtype, tensor_type):
    a = np.arange(6, dtype=np.int64).reshape(2, 3).astype(dtype)
    b = np.arange(6, 12, dtype=np.int64).reshape(2, 3).astype(dtype)
    run_and_compare("Concat", inputs={"A": a, "B": b}, outputs=[("Y", tensor_type)], attrs={"axis": 0})


def test_concat_bfloat16():
    a = float32_to_bfloat16_bits(np.arange(6, dtype=np.float32).reshape(2, 3))
    b = float32_to_bfloat16_bits(np.arange(6, 12, dtype=np.float32).reshape(2, 3))
    model = build_model_with_input_types(
        "Concat", inputs={"A": a, "B": b},
        input_types={"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)], attrs={"axis": 0},
    )
    (actual,) = run_with_iobinding(
        model, {"A": a, "B": b},
        {"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, (4, 3))], use_musa=True,
    )
    np.testing.assert_array_equal(actual, np.concatenate([a, b], axis=0))


def test_concat_bfloat16_unirank_many_small_rows():
    rng = np.random.default_rng(17)
    float_inputs = {
        "X0": rng.standard_normal((1, 3, 256)).astype(np.float32),
        **{
            f"X{i}": rng.standard_normal((1, 3, 128)).astype(np.float32)
            for i in range(1, 8)
        },
    }
    inputs = {
        name: float32_to_bfloat16_bits(value)
        for name, value in float_inputs.items()
    }
    input_types = {name: TensorProto.BFLOAT16 for name in inputs}
    model = build_model_with_input_types(
        "Concat",
        inputs=inputs,
        input_types=input_types,
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"axis": -1},
    )
    (actual,) = run_with_iobinding(
        model,
        inputs,
        input_types,
        [("Y", TensorProto.BFLOAT16, (1, 3, 1152))],
        use_musa=True,
    )
    np.testing.assert_array_equal(
        actual, np.concatenate(list(inputs.values()), axis=-1)
    )


def test_concat_many_small_inputs_axis1():
    inputs = {}
    rng = np.random.default_rng(6)
    for i in range(40):
        width = 1 if i % 2 == 0 else 3
        inputs[f"X{i}"] = rng.standard_normal((8, width)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs=inputs,
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_many_small_inputs_near_direct_kernel_limit_axis1():
    inputs = {}
    rng = np.random.default_rng(16)
    for i in range(237):
        width = 1 if i % 5 else 3
        inputs[f"X{i}"] = rng.standard_normal((2, width)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs=inputs,
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_small_int64_shape_metadata_mixed_inputs():
    nodes = [
        helper.make_node("Constant", [], ["prefix"], value=helper.make_tensor("prefix_value", TensorProto.INT64, [1], [1])),
        helper.make_node("Unsqueeze", ["dynamic_a", "axes"], ["dynamic_a_unsqueezed"]),
        helper.make_node("Unsqueeze", ["dynamic_b", "axes"], ["dynamic_b_unsqueezed"]),
        helper.make_node(
            "Concat",
            ["prefix", "dynamic_a_unsqueezed", "dynamic_b_unsqueezed"],
            ["Y"],
            axis=0,
        ),
    ]
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    model = build_graph_model(
        nodes,
        inputs={
            "dynamic_a": np.array(7, dtype=np.int64),
            "dynamic_b": np.array(11, dtype=np.int64),
        },
        outputs=[("Y", TensorProto.INT64)],
        initializers=[axes],
        opset=17,
        name="concat_shape_metadata_graph",
    )
    run_model_and_compare(model, {"dynamic_a": np.array(7, dtype=np.int64), "dynamic_b": np.array(11, dtype=np.int64)}, rtol=0, atol=0)
