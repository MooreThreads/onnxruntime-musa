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


def test_einsum_opset12_ij_bjk_to_bik_jd_shape():
    lhs = np.arange(32 * 447, dtype=np.float32).reshape(32, 447) / 1024.0
    rhs = np.arange(2 * 447 * 1, dtype=np.float32).reshape(2, 447, 1) / 2048.0
    run_and_compare(
        "Einsum",
        inputs={"lhs": lhs, "rhs": rhs},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "ij,bjk->bik"},
        opset=12,
        rtol=1e-4,
        atol=1e-4,
    )


def test_einsum_opset12_ij_bjk_to_bik_general_k():
    lhs = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) / 7.0
    rhs = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) / 11.0
    run_and_compare(
        "Einsum",
        inputs={"lhs": lhs, "rhs": rhs},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "ij,bjk->bik"},
        opset=12,
        rtol=1e-5,
        atol=1e-5,
    )


def test_einsum_opset12_blhw_bjhw_to_bhl():
    lhs = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5) / 17.0
    rhs = np.arange(2 * 6 * 4 * 5, dtype=np.float32).reshape(2, 6, 4, 5) / 19.0
    run_and_compare(
        "Einsum",
        inputs={"lhs": lhs, "rhs": rhs},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "blhw,bjhw->bhl"},
        opset=12,
        rtol=1e-4,
        atol=1e-4,
    )


def test_einsum_opset12_ilhw_bjhw_to_bhl():
    lhs = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5) / 17.0
    rhs = np.arange(7 * 6 * 4 * 5, dtype=np.float32).reshape(7, 6, 4, 5) / 19.0
    run_and_compare(
        "Einsum",
        inputs={"lhs": lhs, "rhs": rhs},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"equation": "ilhw,bjhw->bhl"},
        opset=12,
        rtol=1e-4,
        atol=1e-4,
    )
