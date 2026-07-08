# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Softplus operator."""

import numpy as np
import pytest

from op_test_utils import (
    TensorProto,
    build_model,
    run,
    run_and_compare,
)


def test_softplus_opset1_float():
    x = np.array([[-1.5, -0.5, 0.5, 1.5, 2.25]], dtype=np.float32)
    run_and_compare(
        "Softplus",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=1,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
        (np.float64, TensorProto.DOUBLE, 1e-6, 1e-7),
    ],
)
def test_softplus_float_like_dtypes(np_dtype, tensor_type, rtol, atol):
    x = np.array([[-1.5, -0.5, 0.5, 1.5, 2.25]], dtype=np_dtype)
    expected = np.log1p(np.exp(-np.abs(x.astype(np.float64)))) + np.maximum(
        x.astype(np.float64), 0.0
    )
    model = build_model(
        "Softplus",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        opset=1,
    )
    (actual,) = run(model, {"X": x}, use_musa=True)
    np.testing.assert_allclose(actual, expected.astype(np_dtype), rtol=rtol, atol=atol)
