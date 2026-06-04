# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Softplus operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_softplus_opset1_float():
    x = np.array([[-1.5, -0.5, 0.5, 1.5, 2.25]], dtype=np.float32)
    run_and_compare(
        "Softplus",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        opset=1,
        rtol=1e-5,
        atol=1e-6,
    )
