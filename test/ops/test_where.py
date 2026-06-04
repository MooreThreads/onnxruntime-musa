# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Where operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_where_float_broadcast_opset16():
    cond = np.array([[True, False, True]], dtype=np.bool_)
    a = np.array([[1.0], [2.0]], dtype=np.float32)
    b = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    run_and_compare(
        "Where",
        inputs={"condition": cond, "X": a, "Y": b},
        outputs=[("Z", TensorProto.FLOAT)],
        opset=16,
    )
