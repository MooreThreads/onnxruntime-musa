# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the LeakyRelu operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_leaky_relu_default_alpha():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare("LeakyRelu", inputs={"X": x}, outputs=[("Y", TensorProto.FLOAT)])


def test_leaky_relu_custom_alpha():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "LeakyRelu",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"alpha": 0.2},
    )
