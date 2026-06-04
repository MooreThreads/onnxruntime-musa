# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the GreaterOrEqual operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_greater_or_equal_float_broadcast_opset16():
    a = np.array([[1.0], [3.0]], dtype=np.float32)
    b = np.array([[2.0, 3.0]], dtype=np.float32)
    run_and_compare(
        "GreaterOrEqual",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        opset=16,
    )
