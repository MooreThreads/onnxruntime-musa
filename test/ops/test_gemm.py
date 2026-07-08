# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Gemm operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    run_and_compare,
    run_with_iobinding,
)


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


def test_gemm_double_with_bias():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((4, 8)).astype(np.float64)
    b = rng.standard_normal((8, 5)).astype(np.float64)
    c = rng.standard_normal((5,)).astype(np.float64)
    run_and_compare(
        "Gemm",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.DOUBLE)],
        attrs={"alpha": 1.25, "beta": -0.5},
        rtol=1e-9,
        atol=1e-10,
    )


def test_gemm_float16_with_bias():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((4, 8)).astype(np.float16)
    b = rng.standard_normal((8, 5)).astype(np.float16)
    c = rng.standard_normal((5,)).astype(np.float16)
    attrs = {"alpha": 0.5, "beta": 0.25}
    expected = (
        attrs["alpha"] * (a.astype(np.float32) @ b.astype(np.float32))
        + attrs["beta"] * c.astype(np.float32)
    ).astype(np.float16)
    model = build_model_with_input_types(
        "Gemm",
        inputs={"A": a, "B": b, "C": c},
        input_types={
            "A": TensorProto.FLOAT16,
            "B": TensorProto.FLOAT16,
            "C": TensorProto.FLOAT16,
        },
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs=attrs,
    )
    (actual,) = run_with_iobinding(
        model,
        {"A": a, "B": b, "C": c},
        {
            "A": TensorProto.FLOAT16,
            "B": TensorProto.FLOAT16,
            "C": TensorProto.FLOAT16,
        },
        [("Y", TensorProto.FLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
