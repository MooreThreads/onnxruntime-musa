# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Sum (variadic) operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_sum_three_inputs():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((2, 3)).astype(np.float32)
    c = rng.standard_normal((2, 3)).astype(np.float32)
    run_and_compare(
        "Sum", inputs={"A": a, "B": b, "C": c}, outputs=[("Y", TensorProto.FLOAT)]
    )


def test_sum_broadcast():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    run_and_compare("Sum", inputs={"A": a, "B": b}, outputs=[("Y", TensorProto.FLOAT)])
