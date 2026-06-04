# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Tile operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_tile_float():
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    repeats = np.array([1, 3], dtype=np.int64)
    run_and_compare("Tile", inputs={"X": x, "repeats": repeats}, outputs=[("Y", TensorProto.FLOAT)])
