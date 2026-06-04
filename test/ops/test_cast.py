# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Cast operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_cast_float_to_int64():
    x = np.random.default_rng(0).uniform(-5.0, 5.0, (16, 32)).astype(np.float32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"to": TensorProto.INT64},
    )


def test_cast_int64_to_float():
    x = np.arange(-256, 256, dtype=np.int64).reshape(16, 32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"to": TensorProto.FLOAT},
    )


def test_cast_float_to_int32():
    x = np.random.default_rng(1).uniform(-5.0, 5.0, (16, 32)).astype(np.float32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"to": TensorProto.INT32},
    )


def test_cast_int32_to_bool():
    x = np.array([[0, 1, -2], [3, 0, 4]], dtype=np.int32)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.BOOL)],
        attrs={"to": TensorProto.BOOL},
    )


def test_cast_bool_to_int64():
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"to": TensorProto.INT64},
    )


def test_cast_int64_to_int32():
    x = np.arange(-12, 12, dtype=np.int64).reshape(4, 6)
    run_and_compare(
        "Cast",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"to": TensorProto.INT32},
    )
