# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the MatMul operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_matmul_2d():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_batched():
    a = np.random.default_rng(2).standard_normal((16, 16, 32)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((16, 32, 64)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_batched_broadcast_left_batch():
    a = np.random.default_rng(4).standard_normal((1, 2, 3)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((4, 3, 5)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_rank4_cross_broadcast():
    a = np.random.default_rng(6).standard_normal((2, 1, 3, 4)).astype(np.float32)
    b = np.random.default_rng(7).standard_normal((1, 5, 4, 6)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)], rtol=2e-3, atol=2e-4)


def test_matmul_rank4_contiguous_batched():
    a = np.random.default_rng(20).standard_normal((2, 3, 4, 8)).astype(np.float32)
    b = np.random.default_rng(21).standard_normal((2, 3, 8, 5)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)], rtol=2e-3, atol=2e-4)


def test_matmul_batched_broadcast_right_matrix():
    a = np.random.default_rng(22).standard_normal((4, 6, 7)).astype(np.float32)
    b = np.random.default_rng(23).standard_normal((7, 3)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)], rtol=2e-3, atol=2e-4)


def test_matmul_double_2d():
    a = np.random.default_rng(8).standard_normal((4, 8)).astype(np.float64)
    b = np.random.default_rng(9).standard_normal((8, 5)).astype(np.float64)
    run_and_compare(
        "MatMul",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.DOUBLE)],
        rtol=1e-9,
        atol=1e-10,
    )


def test_matmul_float16_2d():
    a = np.random.default_rng(10).standard_normal((4, 8)).astype(np.float16)
    b = np.random.default_rng(11).standard_normal((8, 5)).astype(np.float16)
    model = build_model_with_input_types(
        "MatMul",
        inputs={"A": a, "B": b},
        input_types={"A": TensorProto.FLOAT16, "B": TensorProto.FLOAT16},
        outputs=[("Y", TensorProto.FLOAT16)],
    )
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    (actual,) = run_with_iobinding(
        model,
        {"A": a, "B": b},
        {"A": TensorProto.FLOAT16, "B": TensorProto.FLOAT16},
        [("Y", TensorProto.FLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


def test_matmul_bfloat16_2d():
    a_f32 = np.random.default_rng(12).standard_normal((3, 7)).astype(np.float32)
    b_f32 = np.random.default_rng(13).standard_normal((7, 4)).astype(np.float32)
    a = float32_to_bfloat16_bits(a_f32)
    b = float32_to_bfloat16_bits(b_f32)
    expected = bfloat16_bits_to_float32(a) @ bfloat16_bits_to_float32(b)
    model = build_model_with_input_types(
        "MatMul",
        inputs={"A": a, "B": b},
        input_types={"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"A": a, "B": b},
        {"A": TensorProto.BFLOAT16, "B": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual), expected, rtol=3e-2, atol=3e-2
    )


def test_matmul_vector_matrix():
    a = np.random.default_rng(14).standard_normal((8,)).astype(np.float32)
    b = np.random.default_rng(15).standard_normal((8, 5)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_matrix_vector():
    a = np.random.default_rng(16).standard_normal((4, 8)).astype(np.float32)
    b = np.random.default_rng(17).standard_normal((8,)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])


def test_matmul_vector_vector_scalar():
    a = np.random.default_rng(18).standard_normal((8,)).astype(np.float32)
    b = np.random.default_rng(19).standard_normal((8,)).astype(np.float32)
    run_and_compare("MatMul", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
