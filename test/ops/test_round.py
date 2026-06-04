# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Round operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_round_opset11_float():
    x = np.array([[-1.5, -0.5, 0.5, 1.5, 2.25]], dtype=np.float32)
    run_and_compare("Round", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)], opset=11)
