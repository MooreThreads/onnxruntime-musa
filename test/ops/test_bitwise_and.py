# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the BitwiseAnd operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_bitwise_and_int32_opset18():
    a = np.array([[1, 3], [7, 8]], dtype=np.int32)
    b = np.array([[1, 2], [3, 4]], dtype=np.int32)
    run_and_compare(
        "BitwiseAnd",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.INT32)],
        opset=18,
    )
