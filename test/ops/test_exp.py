# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Exp operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_exp_basic():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    run_and_compare("Exp", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_exp_2d():
    x = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
    # Clamp to avoid overflow
    x = np.clip(x, -10.0, 10.0)
    run_and_compare("Exp", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
