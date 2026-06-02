# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the RandomUniformLike operator.

RandomUniformLike generates random values matching the input shape/dtype and
cannot be compared numerically between CPU and MUSA.  The tests verify output
shape, dtype, and value range.
"""

import numpy as np

from op_test_utils import TensorProto, build_model, musa_available, run


def _run_random_uniform_like(input_arr, low=0.0, high=1.0, seed=42.0):
    attrs = {"low": low, "high": high, "seed": seed}
    model_bytes = build_model(
        "RandomUniformLike",
        inputs={"X": input_arr},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs=attrs,
    )
    cpu_out = run(model_bytes, {"X": input_arr}, use_musa=False)
    if musa_available():
        musa_out = run(model_bytes, {"X": input_arr}, use_musa=True)
        assert musa_out[0].shape == input_arr.shape
        assert musa_out[0].dtype == np.float32
        assert np.all(musa_out[0] >= low) and np.all(musa_out[0] < high)
    return cpu_out[0]


def test_random_uniform_like_basic():
    x = np.zeros((4, 8), dtype=np.float32)
    out = _run_random_uniform_like(x)
    assert out.shape == (4, 8)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0) and np.all(out < 1.0)


def test_random_uniform_like_custom_range():
    x = np.ones((10,), dtype=np.float32)
    out = _run_random_uniform_like(x, low=-1.0, high=1.0)
    assert out.shape == (10,)
    assert np.all(out >= -1.0) and np.all(out < 1.0)


def test_random_uniform_like_2d():
    x = np.zeros((3, 5), dtype=np.float32)
    out = _run_random_uniform_like(x, seed=0.0)
    assert out.shape == (3, 5)
