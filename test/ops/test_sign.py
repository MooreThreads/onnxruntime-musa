# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Sign operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_sign_basic():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    run_and_compare("Sign", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_sign_2d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    run_and_compare("Sign", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
