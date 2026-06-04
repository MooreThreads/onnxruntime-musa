# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end MUSA tests for RandomUniform operators."""

import numpy as np

from op_test_utils import TensorProto, build_model, run


def test_random_uniform_opset1_runs_on_musa_without_cpu_fallback():
    model = build_model(
        "RandomUniform",
        inputs={},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "shape": [2, 3],
            "dtype": TensorProto.FLOAT,
            "low": -1.0,
            "high": 2.0,
            "seed": 3.0,
        },
        opset=1,
    )
    (actual,) = run(model, {}, use_musa=True)
    assert actual.shape == (2, 3)
    assert actual.dtype == np.float32
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 2.0)


def test_random_uniform_like_float_opset1_runs_on_musa_without_cpu_fallback():
    x = np.zeros((2, 3), dtype=np.float32)
    model = build_model(
        "RandomUniformLike",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"low": -1.0, "high": 2.0, "seed": 3.0},
        opset=1,
    )
    (actual,) = run(model, {"X": x}, use_musa=True)
    assert actual.shape == x.shape
    assert actual.dtype == np.float32
    assert np.all(actual >= -1.0)
    assert np.all(actual <= 2.0)
