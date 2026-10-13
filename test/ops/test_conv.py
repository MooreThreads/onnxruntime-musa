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
"""End-to-end CPU-vs-MUSA test for the Conv operator."""

import numpy as np
import pytest

from op_test_utils import TensorProto, build_model, run, run_and_compare


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


def test_conv_opset11_float_nchw_with_bias():
    x = np.random.default_rng(5).standard_normal((1, 2, 4, 4)).astype(np.float32)
    w = np.random.default_rng(6).standard_normal((3, 2, 3, 3)).astype(np.float32)
    b = np.array([1.25, -2.0, 0.5], dtype=np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"pads": [1, 1, 1, 1], "strides": [1, 1]},
        opset=11,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv_opset11_float_nchw_grouped_with_bias():
    x = np.random.default_rng(7).standard_normal((1, 4, 5, 5)).astype(np.float32)
    w = np.random.default_rng(8).standard_normal((6, 2, 3, 3)).astype(np.float32)
    b = np.random.default_rng(9).standard_normal((6,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"group": 2, "pads": [1, 1, 1, 1], "strides": [1, 1]},
        opset=11,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv_opset11_float_nchw_depthwise_fallback_with_bias():
    # H=1 and kH=1 deliberately selects Conv's custom device fallback path.
    x = np.random.default_rng(10).standard_normal((1, 4, 1, 7)).astype(np.float32)
    w = np.random.default_rng(11).standard_normal((4, 1, 1, 3)).astype(np.float32)
    b = np.random.default_rng(12).standard_normal((4,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"group": 4, "pads": [0, 1, 0, 1], "strides": [1, 1]},
        opset=11,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv_opset11_float_ncdhw_with_bias():
    x = np.random.default_rng(13).standard_normal((1, 2, 4, 5, 6)).astype(np.float32)
    w = np.random.default_rng(14).standard_normal((3, 2, 3, 3, 3)).astype(np.float32)
    b = np.random.default_rng(15).standard_normal((3,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "pads": [1, 1, 1, 1, 1, 1],
            "strides": [1, 2, 1],
            "dilations": [1, 1, 1],
        },
        opset=11,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv_opset11_float_ncdhw_mudnn_failure_is_explicit(monkeypatch):
    x = np.random.default_rng(16).standard_normal((1, 2, 3, 4, 5)).astype(np.float32)
    w = np.random.default_rng(17).standard_normal((3, 2, 1, 3, 3)).astype(np.float32)
    model = build_model(
        "Conv",
        inputs={"X": x, "W": w},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"pads": [0, 1, 1, 0, 1, 1]},
        opset=11,
    )

    monkeypatch.setenv("ORT_MUSA_CONV3D_FORCE_MUDNN_FAILURE_FOR_TEST", "1")
    with pytest.raises(Exception, match="Conv3D requires the muDNN path"):
        run(model, {"X": x, "W": w}, use_musa=True)


def test_conv_lowered_conv1d_h1_k1():
    x = np.random.default_rng(0).standard_normal((2, 64, 1, 8)).astype(np.float32)
    w = np.random.default_rng(1).standard_normal((16, 64, 1, 1)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 1],
            "strides": [1, 1],
        },
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv_lowered_conv1d_h1_k1_with_bias():
    x = np.random.default_rng(2).standard_normal((2, 8, 1, 5)).astype(np.float32)
    w = np.random.default_rng(3).standard_normal((4, 8, 1, 1)).astype(np.float32)
    b = np.random.default_rng(4).standard_normal((4,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 1],
            "strides": [1, 1],
        },
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv1d_opset13_float_ncl_with_bias_stride_pad():
    x = np.random.default_rng(20).standard_normal((2, 3, 10)).astype(np.float32)
    w = np.random.default_rng(21).standard_normal((4, 3, 3)).astype(np.float32)
    b = np.random.default_rng(22).standard_normal((4,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"pads": [1, 1], "strides": [2], "dilations": [1]},
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv1d_opset13_float_ncl_dilated():
    x = np.random.default_rng(23).standard_normal((1, 2, 12)).astype(np.float32)
    w = np.random.default_rng(24).standard_normal((3, 2, 3)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"pads": [2, 2], "dilations": [2]},
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv1d_opset13_float_ncl_grouped_with_bias():
    x = np.random.default_rng(25).standard_normal((2, 6, 9)).astype(np.float32)
    w = np.random.default_rng(26).standard_normal((9, 2, 4)).astype(np.float32)
    b = np.random.default_rng(27).standard_normal((9,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"group": 3, "pads": [1, 1], "strides": [1]},
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv1d_opset13_float_ncl_depthwise_with_bias():
    # A large grouped kernel mirrors the VITS-style position convolution.
    x = np.random.default_rng(28).standard_normal((1, 16, 20)).astype(np.float32)
    w = np.random.default_rng(29).standard_normal((16, 1, 13)).astype(np.float32)
    b = np.random.default_rng(30).standard_normal((16,)).astype(np.float32)
    run_and_compare(
        "Conv",
        inputs={"X": x, "W": w, "B": b},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"group": 16, "pads": [6, 6], "strides": [1]},
        opset=13,
        rtol=1e-4,
        atol=1e-4,
    )
