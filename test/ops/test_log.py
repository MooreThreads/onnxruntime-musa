# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Log operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_log_float():
    x = np.random.default_rng(0).uniform(0.1, 10.0, (16, 32)).astype(np.float32)
    run_and_compare("Log", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
