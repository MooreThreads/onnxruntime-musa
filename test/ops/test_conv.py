# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Conv operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_conv_opset11_float_nchw():
    x = np.arange(1 * 1 * 5 * 4, dtype=np.float32).reshape(1, 1, 5, 4) / 10.0
    w = np.array([[[[1.0], [0.5], [-1.0]]]], dtype=np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"pads": [1, 0, 1, 0], "strides": [1, 1]},
        opset=11,
        rtol=1e-5,
        atol=1e-5,
    )
