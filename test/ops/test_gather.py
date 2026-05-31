# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Gather operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_gather_axis0():
    data = np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32)
    indices = np.array([0, 8, 16, 24], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_gather_axis1():
    data = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    indices = np.array([[0, 8], [16, 24]], dtype=np.int64)
    run_and_compare(
        "Gather",
        inputs={"data": data, "indices": indices},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )
