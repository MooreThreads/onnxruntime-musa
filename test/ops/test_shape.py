# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Shape operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_shape_float():
    x = np.random.default_rng(0).standard_normal((16, 32, 16)).astype(np.float32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_int64():
    x = np.arange(512, dtype=np.int64).reshape(16, 32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_bool_empty_dim():
    x = np.zeros((0, 3, 1), dtype=np.bool_)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])


def test_shape_int32_scalar():
    x = np.array(7, dtype=np.int32)
    run_and_compare("Shape", inputs={"X": x}, outputs=[("Y", TensorProto.INT64)])
