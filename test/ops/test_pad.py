# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Pad operator (opset 13, mode='constant')."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_pad_1d_both_sides():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pads = np.array([1, 2], dtype=np.int64)
    run_and_compare(
        "Pad",
        inputs={"data": x, "pads": pads},
        outputs=[("output", TensorProto.FLOAT)],
    )


def test_pad_2d_constant():
    x = np.ones((2, 3), dtype=np.float32)
    pads = np.array([0, 1, 0, 1], dtype=np.int64)  # [top, left, bottom, right]
    run_and_compare(
        "Pad",
        inputs={"data": x, "pads": pads},
        outputs=[("output", TensorProto.FLOAT)],
    )


def test_pad_2d_with_value():
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    pads = np.array([1, 2, 1, 2], dtype=np.int64)
    constant_value = np.array([-1.0], dtype=np.float32)
    run_and_compare(
        "Pad",
        inputs={"data": x, "pads": pads, "constant_value": constant_value},
        outputs=[("output", TensorProto.FLOAT)],
    )
