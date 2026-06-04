# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end CPU-vs-MUSA test for the Not operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_not_bool():
    x = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    run_and_compare("Not", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])


def test_not_bool_3d():
    x = np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=np.bool_)
    run_and_compare("Not", inputs={"X": x}, outputs=[("Y", TensorProto.BOOL)])
