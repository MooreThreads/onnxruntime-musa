# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wide-range float32 accuracy checks for the single-precision unary kernels.

The float32 unary device path evaluates the transcendental functions in single
precision (``erff``/``sinf``/``cosf``/``tanhf``/``expf``/...), exactly like the
CUDA EP, instead of promoting every element to double first. These tests pin the
resulting accuracy against the CPU EP over the magnitude ranges where the old
FP64 promotion was strongest: the Erf tail and near-zero region, the Gelu large
positive/negative tails, and the Sin/Cos large-angle range (|x| up to 1e5).

The gate is the ORT default float32 tolerance used by the CUDA EP op tests
(``atol=1e-5`` + ``rtol=1e-4 * |expected|``, numpy.isclose style, see
``onnxruntime/test/providers/checkers.cc`` ``DefaultTolerance<float>``). Each
check prints the measured per-element max abs difference so the number is part
of the test log rather than only implied by a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from op_test_utils import TensorProto, build_model, run

# onnxruntime/test/providers/checkers.cc, DefaultTolerance<float> (inference).
_ATOL = 1e-5
_RTOL = 1e-4


def _check_fp32_gate(op_type, x, opset, attrs=None):
    """Run ``op_type`` on CPU and MUSA and assert the ORT float32 tolerance."""
    x = np.asarray(x, dtype=np.float32)
    model = build_model(
        op_type, {"X": x}, [("Y", TensorProto.FLOAT)], attrs, "", opset
    )
    (cpu,) = run(model, {"X": x}, use_musa=False)
    (musa,) = run(model, {"X": x}, use_musa=True)
    assert cpu.shape == musa.shape == x.shape, f"{op_type}: shape mismatch"

    expected = cpu.astype(np.float64)
    diff = np.abs(musa.astype(np.float64) - expected)
    gate = _ATOL + _RTOL * np.abs(expected)
    ratio = diff / gate
    print(
        f"GATE {op_type:9s} opset{opset:<3d} n={x.size:6d} "
        f"max_abs={diff.max():.3e} max_ratio={ratio.max():.3f}",
        flush=True,
    )
    worst = int(np.argmax(ratio))
    assert ratio.max() <= 1.0, (
        f"{op_type} opset{opset}: element {worst} (x={x.ravel()[worst]:.6g}) "
        f"diff={diff.ravel()[worst]:.3e} exceeds atol={_ATOL} + "
        f"rtol={_RTOL}*|cpu|={gate.ravel()[worst]:.3e}"
    )
    return diff.max()


# The Erf tail saturates at +/-1 from about |x| >= 4 and is ~2x/sqrt(pi) near 0.
_ERF_VALUES = np.array(
    [
        -12.0, -10.0, -8.0, -6.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0,
        -1.0, -0.5, -1e-3, -1e-6, -0.0, 0.0, 1e-6, 1e-3, 0.5, 1.0, 2.0, 2.5,
        3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0,
    ],
    dtype=np.float32,
)


@pytest.mark.parametrize("opset", [11, 13])
def test_erf_float_wide_range(opset):
    tail = np.random.default_rng(7).uniform(-10.0, 10.0, 4096).astype(np.float32)
    near_zero = np.random.default_rng(8).uniform(-1e-4, 1e-4, 4096).astype(np.float32)
    x = np.concatenate([_ERF_VALUES, tail, near_zero]).reshape(1, -1)
    _check_fp32_gate("Erf", x, opset)


def test_gelu_float_large_magnitudes():
    x = np.array(
        [
            -100.0, -50.0, -30.0, -20.0, -12.0, -8.0, -6.0, -4.0, -2.0, -1.0,
            -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0,
            50.0, 100.0,
        ],
        dtype=np.float32,
    )
    _check_fp32_gate("Gelu", x.reshape(1, -1), 20)


@pytest.mark.parametrize("op_type", ["Sin", "Cos"])
@pytest.mark.parametrize("opset", [11, 13])
def test_trig_float_large_angles(op_type, opset):
    fixed = np.array(
        [
            0.0, 1e-6, 1e-3, 0.5, 1.0, 1.5707963, 3.1415927, 6.2831855, 100.0,
            1000.0, 10000.0, 100000.0, -100000.0, -10000.0, -1000.0, -100.0,
            -6.2831855, -3.1415927, -1.5707963, -1.0, -0.5,
        ],
        dtype=np.float32,
    )
    # A dense sweep in the large-angle regime where the argument reduction, not
    # the polynomial, decides the accuracy.
    large = np.random.default_rng(11).uniform(-1e5, 1e5, 8192).astype(np.float32)
    x = np.concatenate([fixed, large]).reshape(1, -1)
    _check_fp32_gate(op_type, x, opset)


# The remaining unary ops share the same single-precision switch. Exp/Log/Sqrt/
# Reciprocal only see their domain, the rest are safe over both signs.
_DOMAIN_POSITIVE = {
    "Sqrt": np.concatenate(
        [
            np.array([0.0, 1e-6, 0.25, 1.0, 2.0, 16.0, 1e6], dtype=np.float32),
            np.random.default_rng(13).uniform(1e-3, 1e3, 2048).astype(np.float32),
        ]
    ),
    "Log": np.concatenate(
        [
            np.array([1e-6, 1e-3, 0.5, 1.0, 2.0, 1e3, 1e6], dtype=np.float32),
            np.random.default_rng(14).uniform(1e-4, 1e4, 2048).astype(np.float32),
        ]
    ),
    "Exp": np.array(
        [-80.0, -20.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 20.0, 80.0], dtype=np.float32
    ),
    "Reciprocal": np.array(
        [1e-6, 1e-3, 0.25, 1.0, 2.0, 1e3, 1e6, -1e-6, -1.0, -1e6], dtype=np.float32
    ),
}


@pytest.mark.parametrize("op_type", ["Sqrt", "Log", "Exp", "Reciprocal"])
def test_unary_domain_float_wide_range(op_type):
    _check_fp32_gate(op_type, _DOMAIN_POSITIVE[op_type].reshape(1, -1), 13)


@pytest.mark.parametrize(
    "op_type",
    ["Abs", "Neg", "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Softplus", "Sign",
     "Ceil", "Floor", "Round"],
)
def test_unary_signed_float_wide_range(op_type):
    x = np.array(
        [
            -100.0, -30.0, -8.0, -1.25, -0.5, -0.0, 0.0, 0.5, 1.25, 8.0, 30.0,
            100.0, 1e-6, -1e-6,
        ],
        dtype=np.float32,
    )
    attrs = {"alpha": 0.2} if op_type == "LeakyRelu" else None
    opset = 11 if op_type == "Round" else 13
    _check_fp32_gate(op_type, x.reshape(1, -1), opset, attrs)
