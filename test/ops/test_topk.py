# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for TopK."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


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
