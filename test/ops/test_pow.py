# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Pow operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_pow_float():
    base = np.random.default_rng(0).uniform(0.1, 3.0, (16, 32)).astype(np.float32)
    exp = np.random.default_rng(1).uniform(-2.0, 2.0, (16, 32)).astype(np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])


def test_pow_broadcast():
    base = np.random.default_rng(2).uniform(0.5, 2.0, (16, 32, 16)).astype(np.float32)
    exp = np.array([2.0], dtype=np.float32)
    run_and_compare("Pow", inputs={"X": base, "Y": exp}, outputs=[("Z", TensorProto.FLOAT)])
