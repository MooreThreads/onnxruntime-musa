# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Erf operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_erf_float():
    x = np.random.default_rng(0).uniform(-2.0, 2.0, (16, 32)).astype(np.float32)
    run_and_compare("Erf", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_erf_float_matrix():
    x = np.array([[-3.0, -1.0, 0.0], [0.5, 1.0, 3.0]], dtype=np.float32)
    run_and_compare("Erf", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)], rtol=1e-4, atol=1e-5)
