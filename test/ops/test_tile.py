# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Tile operator."""

import numpy as np
from onnx import helper
import pytest

from op_test_utils import (
    TensorProto,
    build_graph_model,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_model_and_compare,
    run_with_iobinding,
)


def test_tile_float():
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    repeats = np.array([1, 3], dtype=np.int64)
    run_and_compare("Tile", inputs={"X": x, "repeats": repeats}, outputs=[("Y", TensorProto.FLOAT)])


def test_tile_identity_repeats_after_device_producer():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1024, 432)).astype(np.float32)
    bias = rng.standard_normal((1024, 432)).astype(np.float32)
    repeats = np.array([1, 1], dtype=np.int64)
    feeds = {"X": x, "B": bias, "repeats": repeats}
    model = build_graph_model(
        [
            helper.make_node("Add", ["X", "B"], ["sum"]),
            helper.make_node("Tile", ["sum", "repeats"], ["Y"]),
        ],
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        name="tile_identity_after_device_producer",
    )
    run_model_and_compare(model, feeds)


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "values"),
    [
        (np.bool_, TensorProto.BOOL, [[True], [False]]),
        (np.int32, TensorProto.INT32, [[1], [2]]),
        (np.int64, TensorProto.INT64, [[1], [2]]),
        (np.float16, TensorProto.FLOAT16, [[1.0], [2.0]]),
    ],
)
def test_tile_byte_copy_dtypes(np_dtype, tensor_type, values):
    x = np.array(values, dtype=np_dtype)
    repeats = np.array([1, 3], dtype=np.int64)
    run_and_compare(
        "Tile",
        inputs={"X": x, "repeats": repeats},
        outputs=[("Y", tensor_type)],
    )


def test_tile_bfloat16():
    x_f32 = np.array([[1.0], [2.0]], dtype=np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    repeats = np.array([1, 3], dtype=np.int64)
    expected = np.tile(x, repeats)
    model = build_model_with_input_types(
        "Tile",
        inputs={"X": x, "repeats": repeats},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x, "repeats": repeats},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_array_equal(actual, expected)
