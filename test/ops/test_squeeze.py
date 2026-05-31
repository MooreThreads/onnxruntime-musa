# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Squeeze operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_squeeze_axis0():
    x = np.random.default_rng(0).standard_normal((1, 16, 32)).astype(np.float32)
    axes = np.array([0], dtype=np.int64)
    run_and_compare("Squeeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.FLOAT)])


def test_squeeze_multiple_axes():
    x = np.random.default_rng(1).standard_normal((1, 16, 1, 32)).astype(np.float32)
    axes = np.array([0, 2], dtype=np.int64)
    run_and_compare("Squeeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.FLOAT)])
