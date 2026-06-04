# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Squeeze operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_squeeze_axis0():
    x = np.random.default_rng(0).standard_normal((1, 16, 32)).astype(np.float32)
    axes = np.array([0], dtype=np.int64)
    run_and_compare("Squeeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.FLOAT)])


def test_squeeze_multiple_axes():
    x = np.random.default_rng(1).standard_normal((1, 16, 1, 32)).astype(np.float32)
    axes = np.array([0, 2], dtype=np.int64)
    run_and_compare("Squeeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.FLOAT)])


def test_squeeze_negative_axis_int64():
    x = np.arange(2 * 3, dtype=np.int64).reshape(2, 1, 3)
    axes = np.array([-2], dtype=np.int64)
    run_and_compare("Squeeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.INT64)])


def test_squeeze_bool_all_singleton_axes():
    x = np.array([[[True, False, True]]], dtype=np.bool_)
    axes = np.array([0, 1], dtype=np.int64)
    run_and_compare("Squeeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.BOOL)])


def test_squeeze_uint8():
    x = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    axes = np.array([0], dtype=np.int64)
    run_and_compare("Squeeze", inputs={"X": x, "axes": axes}, outputs=[("Y", TensorProto.UINT8)])
