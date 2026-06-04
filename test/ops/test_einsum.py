# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Einsum operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_einsum_opset12_diagonal():
    x = np.arange(9, dtype=np.float32).reshape(3, 3)
    run_and_compare(
        "Einsum",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "aa->a"},
        opset=12,
    )


def test_einsum_opset12_broadcast_model_equation():
    lhs = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    rhs = np.array([[1.0, 0.5, -1.0], [0.25, 2.0, 1.0]], dtype=np.float32)
    run_and_compare(
        "Einsum",
        inputs={"lhs": lhs, "rhs": rhs},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "bhij,hk->bkij"},
        opset=12,
        rtol=1e-5,
        atol=1e-5,
    )
