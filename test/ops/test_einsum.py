# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Einsum operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_einsum_matmul():
    A = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    B = np.random.default_rng(1).standard_normal((8, 6)).astype(np.float32)
    run_and_compare(
        "Einsum",
        inputs={"A": A, "B": B},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "ij,jk->ik"},
        opset=12,
        rtol=1e-4,
        atol=1e-5,
    )


def test_einsum_transpose():
    A = np.random.default_rng(2).standard_normal((4, 6)).astype(np.float32)
    run_and_compare(
        "Einsum",
        inputs={"A": A},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "ij->ji"},
        opset=12,
    )


def test_einsum_reduce_sum():
    A = np.random.default_rng(3).standard_normal((4, 6)).astype(np.float32)
    run_and_compare(
        "Einsum",
        inputs={"A": A},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "ij->i"},
        opset=12,
        rtol=1e-4,
        atol=1e-5,
    )


def test_einsum_batched_matmul():
    A = np.random.default_rng(4).standard_normal((2, 4, 8)).astype(np.float32)
    B = np.random.default_rng(5).standard_normal((2, 8, 6)).astype(np.float32)
    run_and_compare(
        "Einsum",
        inputs={"A": A, "B": B},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "bij,bjk->bik"},
        opset=12,
        rtol=1e-4,
        atol=1e-5,
    )
