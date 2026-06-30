# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for ScatterND."""

import numpy as np

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_scatter_nd_float_opset13():
    data = np.zeros((3, 4), dtype=np.float32)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    run_and_compare(
        "ScatterND",
        inputs={"data": data, "indices": indices, "updates": updates},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scatter_nd_int64_scalar_updates_opset13():
    data = np.arange(12, dtype=np.int64).reshape(3, 4)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    updates = np.array([100, 200], dtype=np.int64)
    run_and_compare(
        "ScatterND",
        inputs={"data": data, "indices": indices, "updates": updates},
        outputs=[("Y", TensorProto.INT64)],
        opset=13,
        rtol=0,
        atol=0,
    )


def test_scatter_nd_bool_opset13():
    data = np.zeros((2, 3), dtype=np.bool_)
    indices = np.array([[0, 2], [1, 1]], dtype=np.int64)
    updates = np.array([True, True], dtype=np.bool_)
    run_and_compare(
        "ScatterND",
        inputs={"data": data, "indices": indices, "updates": updates},
        outputs=[("Y", TensorProto.BOOL)],
        opset=13,
        rtol=0,
        atol=0,
    )


def test_scatter_nd_bfloat16_bits_opset13():
    data_f32 = np.zeros((3, 2), dtype=np.float32)
    updates_f32 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    data = float32_to_bfloat16_bits(data_f32)
    updates = float32_to_bfloat16_bits(updates_f32)
    indices = np.array([[0], [2]], dtype=np.int64)
    expected = data.copy()
    expected[0] = updates[0]
    expected[2] = updates[1]
    model = build_model_with_input_types(
        "ScatterND",
        inputs={"data": data, "indices": indices, "updates": updates},
        input_types={"data": TensorProto.BFLOAT16, "updates": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        opset=13,
    )
    (actual,) = run_with_iobinding(
        model,
        {"data": data, "indices": indices, "updates": updates},
        {"data": TensorProto.BFLOAT16, "updates": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)


def test_scatter_nd_reduction_add_opset16():
    run_and_compare(
        "ScatterND",
        inputs={
            "data": np.zeros((2, 2), dtype=np.float32),
            "indices": np.array([[0]], dtype=np.int64),
            "updates": np.ones((1, 2), dtype=np.float32),
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"reduction": "add"},
        opset=16,
        rtol=0,
        atol=0,
    )


def test_scatter_nd_float_reduction_add_duplicate_indices_opset16():
    run_and_compare(
        "ScatterND",
        inputs={
            "data": np.zeros((3, 2), dtype=np.float32),
            "indices": np.array([[0], [0], [-1]], dtype=np.int64),
            "updates": np.array(
                [[1.25, 2.5], [3.75, 4.5], [5.0, 6.25]], dtype=np.float32
            ),
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"reduction": "add"},
        opset=16,
        rtol=1e-6,
        atol=1e-6,
    )


def test_scatter_nd_int64_reduction_add_opset16():
    run_and_compare(
        "ScatterND",
        inputs={
            "data": np.array([1, 2, 3], dtype=np.int64),
            "indices": np.array([[1], [1]], dtype=np.int64),
            "updates": np.array([10, 20], dtype=np.int64),
        },
        outputs=[("Y", TensorProto.INT64)],
        attrs={"reduction": "add"},
        opset=16,
        rtol=0,
        atol=0,
    )
