# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Less operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_less_float_broadcast():
    a = np.array([[1.0], [3.0]], dtype=np.float32)
    b = np.array([[2.0, 3.0]], dtype=np.float32)
    run_and_compare("Less", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])
