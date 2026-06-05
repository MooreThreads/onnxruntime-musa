# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the FusedMatMul contrib operator."""

import numpy as np
import pytest

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


def test_fused_matmul_batched_broadcast_left_batch():
    a = np.random.default_rng(8).standard_normal((1, 8, 16)).astype(np.float32)
    b = np.random.default_rng(9).standard_normal((4, 16, 12)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.75},
        domain="com.microsoft",
        rtol=2e-3,
        atol=2e-4,
    )


def test_fused_matmul_batched_transb():
    a = np.random.default_rng(10).standard_normal((2, 3, 8, 16)).astype(np.float32)
    b = np.random.default_rng(11).standard_normal((2, 3, 12, 16)).astype(np.float32)
    run_and_compare(
        "FusedMatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.5, "transB": 1},
        domain="com.microsoft",
        rtol=2e-3,
        atol=2e-4,
    )


def test_fused_matmul_double_alpha_unsupported():
    a = np.random.default_rng(12).standard_normal((4, 8)).astype(np.float64)
    b = np.random.default_rng(13).standard_normal((8, 5)).astype(np.float64)
    with pytest.raises(Exception, match="unsupported MatMul dtype"):
        run_and_compare(
            "FusedMatMul",
            inputs={"A": a, "B": b},
            outputs=[("Y", TensorProto.DOUBLE)],
            attrs={"alpha": 0.25},
            domain="com.microsoft",
            rtol=1e-9,
            atol=1e-10,
        )


def test_fused_matmul_trans_batch_a_unsupported():
    a = np.random.default_rng(14).standard_normal((5, 2, 3)).astype(np.float32)
    b = np.random.default_rng(15).standard_normal((2, 3, 4)).astype(np.float32)
    with pytest.raises(Exception, match="transBatch"):
        run_and_compare(
            "FusedMatMul",
            inputs={"A": a, "B": b},
            outputs=[("Y", TensorProto.FLOAT)],
            attrs={"transBatchA": 1},
            domain="com.microsoft",
            rtol=2e-3,
            atol=2e-4,
        )


def test_fused_matmul_trans_batch_b_unsupported():
    a = np.random.default_rng(16).standard_normal((2, 5, 3)).astype(np.float32)
    b = np.random.default_rng(17).standard_normal((3, 2, 4)).astype(np.float32)
    with pytest.raises(Exception, match="transBatch"):
        run_and_compare(
            "FusedMatMul",
            inputs={"A": a, "B": b},
            outputs=[("Y", TensorProto.FLOAT)],
            attrs={"transBatchB": 1},
            domain="com.microsoft",
            rtol=2e-3,
            atol=2e-4,
        )
