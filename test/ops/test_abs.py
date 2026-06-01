# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Abs operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_abs_float():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Abs", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])
