# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Concat operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_concat_axis0():
    a = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_concat_axis1():
    a = np.random.default_rng(2).standard_normal((32, 16)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((32, 32)).astype(np.float32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_concat_int64_negative_axis():
    a = np.arange(512, dtype=np.int64).reshape(16, 32)
    b = np.arange(512, 1024, dtype=np.int64).reshape(16, 32)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": -1},
    )


def test_concat_int32_three_inputs_axis2():
    a = np.arange(2 * 3 * 1, dtype=np.int32).reshape(2, 3, 1)
    b = np.arange(100, 100 + 2 * 3 * 2, dtype=np.int32).reshape(2, 3, 2)
    c = np.arange(200, 200 + 2 * 3 * 3, dtype=np.int32).reshape(2, 3, 3)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b, "C": c},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axis": 2},
    )


def test_concat_bool_negative_axis():
    a = np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_)
    b = np.array([[[False, True], [True, False]], [[True, True], [False, False]]], dtype=np.bool_)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"axis": -1},
    )


def test_concat_float16_axis1():
    a = np.random.default_rng(4).standard_normal((4, 2)).astype(np.float16)
    b = np.random.default_rng(5).standard_normal((4, 3)).astype(np.float16)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axis": 1},
    )


def test_concat_uint8_axis0():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    b = np.arange(100, 108, dtype=np.uint8).reshape(2, 4)
    run_and_compare(
        "Concat",
        inputs={"A": a, "B": b},
        outputs=[("Y", TensorProto.UINT8)],
        attrs={"axis": 0},
    )
