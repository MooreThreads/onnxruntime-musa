# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA tests for Floor."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_floor_float():
    x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32) * 4
    run_and_compare(
        "Floor",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=17,
        rtol=0,
        atol=0,
    )


def test_floor_scalar_double():
    x = np.array(-1.25, dtype=np.float64)
    run_and_compare(
        "Floor",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        opset=17,
        rtol=0,
        atol=0,
    )
