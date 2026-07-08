# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""Opset 19 smoke tests for tensor device kernels."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_opset19_cast_float_to_int64():
    x = np.random.default_rng(0).uniform(-5.0, 5.0, (4, 8)).astype(np.float32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"to": TensorProto.INT64},
        opset=19,
    )


def test_opset19_expand_uint8():
    x = np.array([1, 2, 3], dtype=np.uint8)
    shape = np.array([2, 1, 3], dtype=np.int64)
    run_and_compare(
        "Expand",
        inputs={"X": x, "shape": shape},
        outputs=[("Y", TensorProto.UINT8)],
        opset=19,
    )


def test_opset19_concat_float16():
    a = np.random.default_rng(1).standard_normal((2, 3)).astype(np.float16)
    b = np.random.default_rng(2).standard_normal((2, 5)).astype(np.float16)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axis": 1},
        opset=19,
    )


def test_opset19_gather_int32():
    data = np.arange(3 * 4, dtype=np.int32).reshape(3, 4)
    indices = np.array([3, 1], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axis": 1},
        opset=19,
    )


def test_opset19_slice_uint16():
    data = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
    starts = np.array([1], dtype=np.int64)
    ends = np.array([5], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes},
        outputs=[("Y", TensorProto.UINT16)],
        opset=19,
    )


def test_opset19_split_bool():
    x = np.array([[True, False, True, False]], dtype=np.bool_)
    split = np.array([1, 3], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.BOOL), ("Y1", TensorProto.BOOL)],
        attrs={"axis": 1},
        opset=19,
    )


def test_opset19_transpose_float16():
    x = np.random.default_rng(3).standard_normal((2, 3, 4)).astype(np.float16)
    run_and_compare(
        "Transpose",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"perm": [2, 0, 1]},
        opset=19,
    )
