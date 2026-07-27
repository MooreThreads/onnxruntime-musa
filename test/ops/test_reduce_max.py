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
"""End-to-end CPU-vs-MUSA test for the ReduceMax operator."""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_graph_model,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_model_and_compare_with_cpu_fallback,
    run_with_iobinding,
)


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "values", "rtol", "atol"),
    [
        (np.uint8, TensorProto.UINT8, [[1, 2, 3], [4, 0, 1]], 0, 0),
        (np.int8, TensorProto.INT8, [[1, -2, 3], [4, 0, -1]], 0, 0),
        (
            np.float16,
            TensorProto.FLOAT16,
            [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
            2e-2,
            2e-2,
        ),
        (
            np.float32,
            TensorProto.FLOAT,
            [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
            1e-5,
            1e-6,
        ),
        (
            np.float64,
            TensorProto.DOUBLE,
            [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
            1e-6,
            1e-7,
        ),
        (np.int32, TensorProto.INT32, [[1, -2, 3], [4, 0, -1]], 0, 0),
        (np.int64, TensorProto.INT64, [[1, -2, 3], [4, 0, -1]], 0, 0),
    ],
)
def test_reduce_max_registered_dtypes(np_dtype, tensor_type, values, rtol, atol):
    x = np.array(values, dtype=np_dtype)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "ReduceMax",
        inputs={"X": x, "axes": axes},
        outputs=[("Y", tensor_type)],
        attrs={"keepdims": 0},
        opset=18,
        rtol=rtol,
        atol=atol,
    )


def test_reduce_max_bfloat16():
    x_f32 = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    axes = np.array([1], dtype=np.int64)
    expected = np.max(bfloat16_bits_to_float32(x), axis=1)
    model = build_model_with_input_types(
        "ReduceMax",
        inputs={"X": x, "axes": axes},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"keepdims": 0},
        opset=18,
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x, "axes": axes},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_reduce_max_accepts_cpu_boundary_input_from_unique_counts():
    x = np.array([3, 1, 3, 2, 2, 2], dtype=np.int64)
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    unique = helper.make_node(
        "Unique",
        ["X"],
        ["unique_values", "unique_indices", "unique_inverse", "counts"],
        sorted=1,
    )
    reduce_max = helper.make_node(
        "ReduceMax",
        ["counts", "axes"],
        ["Y"],
        keepdims=0,
    )
    model = build_graph_model(
        [unique, reduce_max],
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        initializers=[axes],
        opset=18,
        name="unique_counts_reduce_max",
    )
    run_model_and_compare_with_cpu_fallback(model, {"X": x})
