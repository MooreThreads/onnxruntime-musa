# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Gemm operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_gemm_no_trans_with_bias():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((32, 64)).astype(np.float32)
    c = rng.standard_normal((16, 64)).astype(np.float32)
    run_and_compare(
        "Gemm",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
    )


def test_gemm_trans_b_scaled():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((64, 32)).astype(np.float32)
    c = rng.standard_normal((16, 64)).astype(np.float32)
    run_and_compare(
        "Gemm",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 2.0, "beta": 0.5, "transA": 0, "transB": 1},
    )


def test_gemm_trans_a():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((32, 16)).astype(np.float32)
    b = rng.standard_normal((32, 64)).astype(np.float32)
    c = rng.standard_normal((64,)).astype(np.float32)
    run_and_compare(
        "Gemm",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 1.0, "beta": 1.0, "transA": 1, "transB": 0},
    )


def test_gemm_no_bias_scaled():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    run_and_compare(
        "Gemm",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.25, "beta": 1.0, "transA": 0, "transB": 0},
    )


def test_gemm_column_bias_broadcast():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((16, 12)).astype(np.float32)
    c = rng.standard_normal((12,)).astype(np.float32)
    run_and_compare(
        "Gemm",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 1.0, "beta": -0.5, "transA": 0, "transB": 0},
    )
