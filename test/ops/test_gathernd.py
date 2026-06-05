# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for GatherND."""

import numpy as np

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_gathernd_float_opset13():
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    indices = np.array([[[0, 1], [1, 2]], [[1, 0], [0, 2]]], dtype=np.int64)
    run_and_compare(
        "GatherND",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )


def test_gathernd_batch_dims_int64_opset13():
    data = np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4)
    indices = np.array([[[0], [2]], [[1], [0]]], dtype=np.int64)
    run_and_compare(
        "GatherND",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"batch_dims": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


def test_gathernd_bool_opset13():
    data = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    indices = np.array([[0, 2], [1, 1]], dtype=np.int64)
    run_and_compare(
        "GatherND",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.BOOL)],
        opset=13,
        rtol=0,
        atol=0,
    )


def test_gathernd_bfloat16_bits_opset13():
    data_f32 = np.arange(12, dtype=np.float32).reshape(3, 4) / np.float32(3.0)
    data = float32_to_bfloat16_bits(data_f32)
    indices = np.array([[0, 0], [2, 3], [1, 2]], dtype=np.int64)
    expected = data[[0, 2, 1], [0, 3, 2]]
    model = build_model_with_input_types(
        "GatherND",
        inputs={"data": data, "indices": indices},
        input_types={"data": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        opset=13,
    )
    (actual,) = run_with_iobinding(
        model,
        {"data": data, "indices": indices},
        {"data": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)
