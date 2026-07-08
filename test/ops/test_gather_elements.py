# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for GatherElements."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_gather_elements_float_axis1_int64_indices():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([[0, 2], [1, 0]], dtype=np.int64)
    run_and_compare(
        "GatherElements",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
        opset=17,
        rtol=0,
        atol=0,
    )


def test_gather_elements_int32_axis0_negative_indices():
    data = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    indices = np.array([[0, -1, 0], [1, 0, -1]], dtype=np.int32)
    run_and_compare(
        "GatherElements",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axis": 0},
        opset=17,
        rtol=0,
        atol=0,
    )
