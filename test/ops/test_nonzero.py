# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the NonZero operator."""

import numpy as np
import pytest

from op_test_utils import TensorProto, run_and_compare


def test_nonzero_float():
    x = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    run_and_compare("NonZero", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


@pytest.mark.parametrize(
    ("np_dtype", "values"),
    [
        (np.bool_, [[False, True], [True, False]]),
        (np.uint8, [[0, 2], [3, 0]]),
        (np.int32, [[0, 2], [3, 0]]),
        (np.int64, [[0, 2], [3, 0]]),
        (np.float16, [[0.0, 2.0], [3.0, 0.0]]),
    ],
)
def test_nonzero_registered_dtypes(np_dtype, values):
    x = np.array(values, dtype=np_dtype)
    run_and_compare("NonZero", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_nonzero_multi_block_prefix_sum():
    x = np.zeros((1024,), dtype=np.float32)
    x[1::3] = np.arange(1, x[1::3].size + 1, dtype=np.float32)
    run_and_compare("NonZero", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])
