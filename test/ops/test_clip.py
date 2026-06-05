# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Clip operator."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "values", "min_value", "max_value", "rtol", "atol"),
    [
        (np.float16, TensorProto.FLOAT16, [-2.0, -0.5, 0.5, 2.0], -1.0, 1.0, 2e-2, 2e-2),
        (np.float32, TensorProto.FLOAT, [-2.0, -0.5, 0.5, 2.0], -1.0, 1.0, 1e-5, 1e-6),
        (np.float64, TensorProto.DOUBLE, [-2.0, -0.5, 0.5, 2.0], -1.0, 1.0, 1e-12, 1e-12),
        (np.int8, TensorProto.INT8, [-4, -1, 0, 3], -1, 2, 0, 0),
        (np.uint8, TensorProto.UINT8, [0, 1, 2, 5], 1, 3, 0, 0),
        (np.int64, TensorProto.INT64, [-4, -1, 0, 3], -1, 2, 0, 0),
        (np.uint64, TensorProto.UINT64, [0, 1, 2, 5], 1, 3, 0, 0),
    ],
)
def test_clip_registered_dtypes(
    np_dtype, tensor_type, values, min_value, max_value, rtol, atol
):
    x = np.array(values, dtype=np_dtype)
    min_v = np.array(min_value, dtype=np_dtype)
    max_v = np.array(max_value, dtype=np_dtype)
    run_and_compare(
        "Clip",
        inputs={"X": x, "min": min_v, "max": max_v},
        outputs=[("Y", tensor_type)],
        rtol=rtol,
        atol=atol,
    )


def test_clip_int64_min_only():
    x = np.array([-4, -1, 0, 3], dtype=np.int64)
    min_v = np.array(0, dtype=np.int64)
    run_and_compare(
        "Clip",
        inputs={"X": x, "min": min_v},
        outputs=[("Y", TensorProto.INT64)],
    )
