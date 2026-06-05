# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the GlobalAveragePool operator."""

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
def test_global_average_pool_3d_opset17(np_dtype, tensor_type, rtol, atol):
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4).astype(np_dtype)
    run_and_compare(
        "GlobalAveragePool",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        opset=17,
        rtol=rtol,
        atol=atol,
    )


def test_global_average_pool_double_opset17():
    x = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    expected = np.mean(x, axis=2, keepdims=True)
    model = build_model(
        "GlobalAveragePool",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        opset=17,
    )
    (actual,) = run(model, {"X": x}, use_musa=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_global_average_pool_4d_opset1_float():
    x = np.random.default_rng(0).standard_normal((2, 3, 4, 5)).astype(np.float32)
    run_and_compare(
        "GlobalAveragePool",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=1,
        rtol=1e-5,
        atol=1e-6,
    )
