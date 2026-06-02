# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the RandomUniform operator.

RandomUniform generates random values and cannot be compared numerically
between CPU and MUSA.  The tests verify output shape, dtype, and value range.
"""

import numpy as np

from op_test_utils import TensorProto, build_model, musa_available, run


def _run_random_uniform(shape, low=0.0, high=1.0, seed=42.0):
    attrs = {"shape": shape, "low": low, "high": high, "seed": seed}
    model_bytes = build_model(
        "RandomUniform",
        inputs={},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs=attrs,
    )
    cpu_out = run(model_bytes, {}, use_musa=False)
    if musa_available():
        musa_out = run(model_bytes, {}, use_musa=True)
        assert musa_out[0].shape == tuple(shape)
        assert musa_out[0].dtype == np.float32
        assert np.all(musa_out[0] >= low) and np.all(musa_out[0] < high)
    return cpu_out[0]


def test_random_uniform_basic():
    out = _run_random_uniform([4, 8])
    assert out.shape == (4, 8)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0) and np.all(out < 1.0)


def test_random_uniform_custom_range():
    out = _run_random_uniform([16], low=-2.0, high=2.0)
    assert out.shape == (16,)
    assert np.all(out >= -2.0) and np.all(out < 2.0)


def test_random_uniform_3d():
    out = _run_random_uniform([2, 3, 4], seed=0.0)
    assert out.shape == (2, 3, 4)
