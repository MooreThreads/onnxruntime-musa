# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Gather operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_gather_axis0():
    data = np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32)
    indices = np.array([0, 8, 16, 24], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_gather_axis1():
    data = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    indices = np.array([[0, 8], [16, 24]], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_gather_int32_indices_negative_axis():
    data = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
    indices = np.array([[3, 1], [0, 2]], dtype=np.int32)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axis": -1},
    )


def test_gather_negative_indices():
    data = np.arange(3 * 4 * 2, dtype=np.int64).reshape(3, 4, 2)
    indices = np.array([-1, 0, -3], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axis": 1},
    )


def test_gather_bool_scalar_index():
    data = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    indices = np.array(1, dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"axis": 0},
    )
