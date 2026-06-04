# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Pad operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_pad_constant_float():
    pads = np.array([0, 1, 1, 0], dtype=np.int64)
    value = np.array(5.0, dtype=np.float32)
    run_and_compare(
        "Pad",
        inputs={
            "data": np.arange(4, dtype=np.float32).reshape(2, 2),
            "pads": pads,
            "constant_value": value,
        },
        outputs=[("output", TensorProto.FLOAT)],
        attrs={"mode": "constant"},
    )
