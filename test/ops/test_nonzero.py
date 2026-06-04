# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the NonZero operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_nonzero_float():
    x = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    run_and_compare("NonZero", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])
