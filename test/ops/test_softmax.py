# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Softmax operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_softmax_last_axis():
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "Softmax",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": -1},
    )


def test_softmax_axis0():
    x = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    run_and_compare(
        "Softmax",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )
