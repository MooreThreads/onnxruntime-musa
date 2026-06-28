# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for CumSum."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_cumsum_int32_axis0_opset17():
    x = np.array([2, 3, 5, 7, 11], dtype=np.int32)
    axis = np.array([0], dtype=np.int64)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.INT32)],
        opset=17,
        rtol=0,
        atol=0,
    )


def test_cumsum_float_axis1_exclusive_reverse_opset17():
    x = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float32)
    axis = np.array(1, dtype=np.int32)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"exclusive": 1, "reverse": 1},
        opset=17,
        rtol=1e-5,
        atol=1e-6,
    )


def test_cumsum_int64_negative_axis_opset17():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    axis = np.array([-1], dtype=np.int64)
    run_and_compare(
        "CumSum",
        inputs={"X": x, "axis": axis},
        outputs=[("Y", TensorProto.INT64)],
        opset=17,
        rtol=0,
        atol=0,
    )
