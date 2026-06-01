# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Or operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_or_bool_broadcast():
    a = np.array([[True], [False]], dtype=np.bool_)
    b = np.array([[False, True, False]], dtype=np.bool_)
    run_and_compare("Or", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.BOOL)])
