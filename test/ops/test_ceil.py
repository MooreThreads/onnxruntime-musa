# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Ceil operator."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float32, TensorProto.FLOAT, 1e-5, 1e-6),
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
        (np.float64, TensorProto.DOUBLE, 1e-12, 1e-12),
    ],
)
def test_ceil_float_like(np_dtype, tensor_type, rtol, atol):
    x = np.array([[-1.5, -0.1, 0.0], [0.1, 1.2, 2.0]], dtype=np_dtype)
    run_and_compare(
        "Ceil",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        rtol=rtol,
        atol=atol,
    )
