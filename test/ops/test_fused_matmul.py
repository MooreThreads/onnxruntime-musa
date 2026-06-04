# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the FusedMatMul contrib operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_fused_matmul_2d():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 1.0},
        domain="com.microsoft",
    )


def test_fused_matmul_alpha_transb():
    a = np.random.default_rng(2).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((64, 32)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.5, "transA": 0, "transB": 1},
        domain="com.microsoft",
    )


def test_fused_matmul_transa():
    a = np.random.default_rng(4).standard_normal((32, 16)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((32, 24)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 1.0, "transA": 1, "transB": 0},
        domain="com.microsoft",
    )


def test_fused_matmul_batched_no_transpose():
    a = np.random.default_rng(6).standard_normal((2, 8, 16)).astype(np.float32)
    b = np.random.default_rng(7).standard_normal((2, 16, 12)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 1.0},
        domain="com.microsoft",
    )
