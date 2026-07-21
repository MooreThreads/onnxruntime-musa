# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for TopK."""

import numpy as np
import pytest

from op_test_utils import TensorProto, build_model, run, run_and_compare


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "rtol", "atol"),
    [
        (np.float32, TensorProto.FLOAT, 1e-5, 1e-6),
        (np.float16, TensorProto.FLOAT16, 2e-2, 2e-2),
        (np.float64, TensorProto.DOUBLE, 1e-9, 1e-10),
        (np.int32, TensorProto.INT32, 0, 0),
        (np.int64, TensorProto.INT64, 0, 0),
    ],
)
def test_topk_largest_axis_last_opset13(np_dtype, tensor_type, rtol, atol):
    x = np.array(
        [[[1, 5, 3, 2], [4, 4, 7, 0]], [[9, 8, 8, 1], [3, 6, 2, 5]]],
        dtype=np_dtype,
    )
    k = np.array([2], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", tensor_type), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 1, "sorted": 1},
        opset=13,
        rtol=rtol,
        atol=atol,
    )


def test_topk_smallest_axis1_opset13():
    x = np.array(
        [[[8.0, 1.0], [3.0, 4.0], [2.0, 7.0], [6.0, 5.0]]],
        dtype=np.float32,
    )
    k = np.array([3], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": 1, "largest": 0, "sorted": 1},
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )


def test_topk_large_last_axis_k192_opset13():
    rng = np.random.default_rng(20260703)
    x = rng.uniform(-1.0, 1.0, size=(1, 20000)).astype(np.float32)
    k = np.array([192], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 1, "sorted": 1},
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )


def test_topk_full_last_axis_k_equals_dim_opset13():
    rng = np.random.default_rng(20260709)
    x = rng.uniform(-1.0, 1.0, size=(3, 192)).astype(np.float32)
    k = np.array([192], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 1, "sorted": 1},
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )


def test_topk_smallest_full_axis_real_token_merger_opset13():
    x = np.array(
        [
            [
                0.0,
                1.0,
                9.8013570e06,
                1.0715436e07,
                1.0715513e07,
                1.0715654e07,
                1.0819043e07,
                1.0823217e07,
                0.0,
                1.0,
                0.0,
                1.0715497e07,
                1.0715646e07,
                1.0849186e07,
            ]
        ],
        dtype=np.float32,
    )
    k = np.array([14], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 0, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


def test_topk_int64_small_dim_large_k_opset13():
    rng = np.random.default_rng(20260710)
    x = rng.integers(-10000, 10000, size=(2, 1024), dtype=np.int64)
    k = np.array([512], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.INT64), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 1, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


def test_topk_int64_smallest_small_dim_opset13():
    x = np.array([[5, -1, -1, 9, 0, -3, -3, 7]], dtype=np.int64)
    k = np.array([4], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.INT64), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 0, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


def test_topk_int64_small_dim_axis1_inner_stride_opset13():
    x = np.array(
        [
            [[4, 9, 1], [4, 2, 7], [3, 9, 8], [0, 5, 8], [6, 1, 2]],
            [[-1, 3, 3], [5, 3, 0], [5, 4, 0], [2, 8, 9], [2, 8, 1]],
        ],
        dtype=np.int64,
    )
    k = np.array([3], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.INT64), ("Indices", TensorProto.INT64)],
        attrs={"axis": 1, "largest": 1, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


def test_topk_int64_large_dim_fallback_opset13():
    rng = np.random.default_rng(20260711)
    x = rng.integers(-2000, 2000, size=(1, 1300), dtype=np.int64)
    x[0, 37] = 5000
    x[0, 901] = 5000
    k = np.array([7], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.INT64), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 1, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


def test_topk_k_zero_empty_output_opset13():
    x = np.array([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]], dtype=np.float32)
    k = np.array([0], dtype=np.int64)
    values, indices = run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": 1, "sorted": 1},
        opset=13,
        rtol=1e-5,
        atol=1e-6,
    )
    assert values.shape == (2, 0)
    assert indices.shape == (2, 0)

@pytest.mark.parametrize("largest", [0, 1])
def test_topk_twin_fast_path_all_equal_lower_indices_opset13(largest):
    """The twin shape must preserve the schema lower-index tie-break."""
    x = np.full((1, 4000), 3.25, dtype=np.float32)
    k = np.array([128], dtype=np.int64)
    values, indices = run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": largest, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )

    np.testing.assert_array_equal(values, np.full((1, 128), 3.25, dtype=np.float32))
    np.testing.assert_array_equal(indices, np.arange(128, dtype=np.int64)[None, :])


@pytest.mark.parametrize("largest", [0, 1])
def test_topk_twin_fast_path_repeated_signed_values_opset13(largest):
    """Exercise many positive/negative ties on the exact twin fast-path guard."""
    pattern = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.0])
    x = np.tile(pattern, 500).astype(np.float32)[None, :]
    k = np.array([128], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": largest, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("axis", [1, -2])
@pytest.mark.parametrize("k_value", [1, 3])
def test_topk_sorted1_ties_positive_and_negative_axis_opset13(axis, k_value):
    """Cover positive/negative axes and k=1/small-k with sorted ties."""
    x = ((np.arange(40).reshape(2, 5, 4) % 7) - 3).astype(np.float32)
    k = np.array([k_value], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": axis, "largest": 1, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("largest", [0, 1])
def test_topk_sorted0_valid_set_and_value_index_pairs_opset13(largest):
    """For sorted=0 compare selected pairs without requiring output order."""
    x = np.array(
        [
            [5.0, 5.0, 5.0, 4.0, 4.0, 1.0, 0.0, -1.0, -1.0, -2.0, -2.0, -3.0],
            [2.0, -4.0, 2.0, -4.0, 3.0, 3.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    k = np.array([4], dtype=np.int64)
    model = build_model(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", TensorProto.FLOAT), ("Indices", TensorProto.INT64)],
        attrs={"axis": -1, "largest": largest, "sorted": 0},
        opset=13,
    )

    cpu_values, _ = run(model, {"X": x, "K": k}, use_musa=False)
    musa_values, musa_indices = run(model, {"X": x, "K": k}, use_musa=True)

    # sorted=0 leaves output order unspecified; compare the selected value set.
    np.testing.assert_allclose(
        np.sort(musa_values, axis=-1),
        np.sort(cpu_values, axis=-1),
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_array_equal(
        musa_values, np.take_along_axis(x, musa_indices, axis=-1)
    )
    assert np.all(musa_indices >= 0)
    assert np.all(musa_indices < x.shape[-1])


@pytest.mark.parametrize(
    ("np_dtype", "shape", "axis", "k_value", "tensor_type"),
    [
        (np.float32, (1, 3999), -1, 128, TensorProto.FLOAT),
        (np.float32, (1, 4000), -1, 127, TensorProto.FLOAT),
        (np.float32, (1, 4000, 2), 1, 128, TensorProto.FLOAT),
        (np.float64, (1, 4000), -1, 128, TensorProto.DOUBLE),
    ],
    ids=["dim", "k", "inner", "dtype"],
)
def test_topk_twin_fast_path_guard_fallback_opset13(
    np_dtype, shape, axis, k_value, tensor_type
):
    """Every near-miss of the strict twin guard must use the generic fallback."""
    rng = np.random.default_rng(20260720)
    x = rng.integers(-16, 17, size=shape).astype(np_dtype)
    k = np.array([k_value], dtype=np.int64)
    run_and_compare(
        "TopK",
        inputs={"X": x, "K": k},
        outputs=[("Values", tensor_type), ("Indices", TensorProto.INT64)],
        attrs={"axis": axis, "largest": 1, "sorted": 1},
        opset=13,
        rtol=0,
        atol=0,
    )
