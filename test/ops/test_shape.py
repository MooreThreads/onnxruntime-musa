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
"""End-to-end CPU-vs-MUSA test for the Shape operator."""

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
    run_model_and_compare_with_cpu_fallback,
    run_with_iobinding,
)


_FIXED_DTYPES_NO_BFLOAT16 = [
    (np.uint8, TensorProto.UINT8),
    (np.uint16, TensorProto.UINT16),
    (np.uint32, TensorProto.UINT32),
    (np.uint64, TensorProto.UINT64),
    (np.int8, TensorProto.INT8),
    (np.int16, TensorProto.INT16),
    (np.int32, TensorProto.INT32),
    (np.int64, TensorProto.INT64),
    (np.float16, TensorProto.FLOAT16),
    (np.float32, TensorProto.FLOAT),
    (np.float64, TensorProto.DOUBLE),
    (np.bool_, TensorProto.BOOL),
]


def _values(dtype):
    if dtype == np.bool_:
        return np.zeros((2, 3, 4), dtype=np.bool_)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.arange(24, dtype=dtype).reshape(2, 3, 4)
    if np.issubdtype(dtype, np.integer):
        return (np.arange(24).reshape(2, 3, 4) - 12).astype(dtype)
    return np.linspace(-1.5, 1.5, 24, dtype=dtype).reshape(2, 3, 4)


def test_shape_float():
    x = np.random.default_rng(0).standard_normal((16, 32, 16)).astype(np.float32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_int64():
    x = np.arange(512, dtype=np.int64).reshape(16, 32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_bool_empty_dim():
    x = np.zeros((0, 3, 1), dtype=np.bool_)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_int32_scalar():
    x = np.array(7, dtype=np.int32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_uint8():
    x = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


@pytest.mark.parametrize(("np_dtype", "tensor_type"), _FIXED_DTYPES_NO_BFLOAT16)
def test_shape_fixed_size_dtypes(np_dtype, tensor_type):
    x = _values(np_dtype)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_bfloat16():
    x = float32_to_bfloat16_bits(
        np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(2, 3, 4)
    )
    model = build_model_with_input_types(
        "Shape",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.INT64)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.INT64, (3,))],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, np.array([2, 3, 4], dtype=np.int64))


def test_shape_cpu_metadata_feeds_cast_gather_indices():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    data = np.arange(10, dtype=np.float32)
    nodes = [
        helper.make_node("Shape", ["X"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["indices"], to=TensorProto.INT32),
        helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0),
    ]
    model = build_graph_model(
        nodes,
        inputs={"X": x, "data": data},
        outputs=[("Y", TensorProto.FLOAT)],
        name="shape_cast_gather_indices",
    )
    run_model_and_compare_with_cpu_fallback(model, {"X": x, "data": data})
