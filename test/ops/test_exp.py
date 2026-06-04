# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Exp operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_exp_float():
    x = np.array([[-1.0, 0.0, 1.0, 3.0]], dtype=np.float32)
    run_and_compare("Exp", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
