# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Transpose operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_transpose_default():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("Transpose", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_transpose_perm():
    x = np.random.default_rng(1).standard_normal((16, 32, 16)).astype(np.float32)
    run_and_compare(
        "Transpose",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"perm": [1, 2, 0]},
    )
