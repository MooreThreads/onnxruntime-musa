# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Round operator."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


def test_round_opset11_float():
    x = np.array([[-1.5, -0.5, 0.5, 1.5, 2.25]], dtype=np.float32)
    run_and_compare("Round", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)], opset=11)


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float16, TensorProto.FLOAT16, 0, 0),
        (np.float64, TensorProto.DOUBLE, 0, 0),
    ],
)
def test_round_float_like_dtypes(np_dtype, tensor_type, rtol, atol):
    x = np.array([[-1.5, -0.5, 0.5, 1.5, 2.25]], dtype=np_dtype)
    run_and_compare(
        "Round",
        inputs={"X": x},
        outputs=[("Y", tensor_type)],
        opset=11,
        rtol=rtol,
        atol=atol,
    )
