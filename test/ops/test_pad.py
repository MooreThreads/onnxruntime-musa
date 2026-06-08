# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Pad operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_pad_constant_float():
    pads = np.array([0, 1, 1, 0], dtype=np.int64)
    value = np.array(5.0, dtype=np.float32)
    run_and_compare(
        "Pad",
        inputs={
            "data": np.arange(4, dtype=np.float32).reshape(2, 2),
            "pads": pads,
            "constant_value": value,
        },
        outputs=[("output", TensorProto.FLOAT)],
        attrs={"mode": "constant"},
    )


def test_pad_zero_right_last_axis_2d():
    data = np.random.default_rng(0).normal(size=(4, 7)).astype(np.float32)
    pads = np.array([0, 0, 0, 3], dtype=np.int64)
    run_and_compare(
        "Pad",
        inputs={"data": data, "pads": pads},
        outputs=[("output", TensorProto.FLOAT)],
        attrs={"mode": "constant"},
    )


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "data", "value"),
    [
        (np.bool_, TensorProto.BOOL, [[True, False], [False, True]], True),
        (np.int32, TensorProto.INT32, [[1, 2], [3, 4]], 5),
        (np.int64, TensorProto.INT64, [[1, 2], [3, 4]], 5),
        (np.float16, TensorProto.FLOAT16, [[1.0, 2.0], [3.0, 4.0]], 5.0),
    ],
)
def test_pad_byte_copy_dtypes(np_dtype, tensor_type, data, value):
    pads = np.array([0, 1, 1, 0], dtype=np.int64)
    run_and_compare(
        "Pad",
        inputs={
            "data": np.array(data, dtype=np_dtype),
            "pads": pads,
            "constant_value": np.array(value, dtype=np_dtype),
        },
        outputs=[("output", tensor_type)],
        attrs={"mode": "constant"},
    )


def test_pad_bfloat16():
    data = float32_to_bfloat16_bits(np.arange(4, dtype=np.float32).reshape(2, 2))
    fill = float32_to_bfloat16_bits(np.array([5.0], dtype=np.float32))
    pads = np.array([0, 1, 1, 0], dtype=np.int64)
    expected = np.pad(data, ((0, 1), (1, 0)), constant_values=int(fill[0]))
    model = build_model_with_input_types(
        "Pad",
        inputs={"data": data, "pads": pads, "constant_value": fill},
        input_types={
            "data": TensorProto.BFLOAT16,
            "constant_value": TensorProto.BFLOAT16,
        },
        outputs=[("output", TensorProto.BFLOAT16)],
        attrs={"mode": "constant"},
    )
    (actual,) = run_with_iobinding(
        model,
        {"data": data, "pads": pads, "constant_value": fill},
        {
            "data": TensorProto.BFLOAT16,
            "constant_value": TensorProto.BFLOAT16,
        },
        [("output", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)
