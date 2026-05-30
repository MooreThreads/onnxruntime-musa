# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the FusedMatMul contrib operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_fused_matmul_2d():
    a = np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((3, 5)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 1.0},
        domain="com.microsoft",
    )


def test_fused_matmul_alpha_transb():
    a = np.random.default_rng(2).standard_normal((4, 3)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((5, 3)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.5, "transA": 0, "transB": 1},
        domain="com.microsoft",
    )
