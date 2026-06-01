# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the BatchNormalization operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_batch_normalization_float_2d():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 4)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, (4,)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, (4,)).astype(np.float32)
    mean = rng.standard_normal((4,)).astype(np.float32)
    var = rng.uniform(0.5, 2.0, (4,)).astype(np.float32)
    run_and_compare(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"epsilon": 1e-5},
        opset=15,
    )


def test_batch_normalization_float_4d():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, (3,)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, (3,)).astype(np.float32)
    mean = rng.standard_normal((3,)).astype(np.float32)
    var = rng.uniform(0.5, 2.0, (3,)).astype(np.float32)
    run_and_compare(
        "BatchNormalization",
        inputs={
            "X": x,
            "scale": scale,
            "B": bias,
            "input_mean": mean,
            "input_var": var,
        },
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"epsilon": 1e-5},
        opset=15,
    )
