# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Split operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_split_even_axis0():
    x = np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32)
    split = np.array([16, 16], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_split_uneven_axis1():
    x = np.random.default_rng(1).standard_normal((32, 48)).astype(np.float32)
    split = np.array([16, 32], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_split_negative_axis_int32():
    x = np.arange(2 * 3 * 5, dtype=np.int32).reshape(2, 3, 5)
    split = np.array([2, 3], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.INT32), ("Y1", TensorProto.INT32)],
        attrs={"axis": -1},
    )


def test_split_bool_three_outputs():
    x = np.array([[True, False, True, False, True, False]], dtype=np.bool_)
    split = np.array([1, 2, 3], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[
            ("Y0", TensorProto.BOOL),
            ("Y1", TensorProto.BOOL),
            ("Y2", TensorProto.BOOL),
        ],
        attrs={"axis": 1},
    )


def test_split_float16_axis0():
    x = np.random.default_rng(2).standard_normal((6, 4)).astype(np.float16)
    split = np.array([2, 4], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT16), ("Y1", TensorProto.FLOAT16)],
        attrs={"axis": 0},
    )


def test_split_uint8_three_outputs():
    x = np.arange(2 * 6, dtype=np.uint8).reshape(2, 6)
    split = np.array([1, 2, 3], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[
            ("Y0", TensorProto.UINT8),
            ("Y1", TensorProto.UINT8),
            ("Y2", TensorProto.UINT8),
        ],
        attrs={"axis": 1},
    )


def test_split_many_single_column_outputs():
    output_count = 64
    x = np.arange(5 * output_count, dtype=np.float32).reshape(5, output_count)
    split = np.ones(output_count, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.FLOAT) for i in range(output_count)]
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=outputs,
        attrs={"axis": 1},
    )
