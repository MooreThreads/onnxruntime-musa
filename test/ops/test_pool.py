# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for MaxPool and GlobalMaxPool."""

import numpy as np
import pytest

from op_test_utils import TensorProto, build_model, run, run_and_compare


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float32, TensorProto.FLOAT, 1e-5, 1e-6),
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
    ],
)
def test_global_max_pool_opset13(np_dtype, tensor_type, rtol, atol):
    x = np.random.default_rng(0).standard_normal((2, 3, 4, 5)).astype(np_dtype)
    run_and_compare(
        "GlobalMaxPool",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        opset=13,
        rtol=rtol,
        atol=atol,
    )


def test_global_max_pool_double_opset13():
    x = np.random.default_rng(0).standard_normal((2, 3, 4, 5)).astype(np.float64)
    expected = np.max(x, axis=(2, 3), keepdims=True)
    model = build_model(
        "GlobalMaxPool",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        opset=13,
    )
    (actual,) = run(model, {"X": x}, use_musa=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_max_pool_2d_float_with_indices_opset13():
    x = np.arange(1 * 2 * 4 * 5, dtype=np.float32).reshape(1, 2, 4, 5)
    run_and_compare(
        "MaxPool",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"kernel_shape": [2, 2], "strides": [2, 1]},
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )


def test_max_pool_2d_pads_dilations_opset13():
    x = np.random.default_rng(1).standard_normal((1, 1, 5, 5)).astype(np.float32)
    run_and_compare(
        "MaxPool",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={
            "kernel_shape": [2, 2],
            "strides": [1, 1],
            "pads": [1, 1, 1, 1],
            "dilations": [2, 2],
        },
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type"),
    [(np.int8, TensorProto.INT8), (np.uint8, TensorProto.UINT8)],
)
def test_max_pool_int8_uint8_opset13(np_dtype, tensor_type):
    x = np.arange(1 * 1 * 4 * 4, dtype=np.int16).reshape(1, 1, 4, 4)
    x = (x - 8).astype(np_dtype)
    run_and_compare(
        "MaxPool",
        inputs={"X": x},
        outputs=[("Y", tensor_type), ("Indices", TensorProto.INT64)],
        attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        opset=13,
        rtol=0,
        atol=0,
    )
