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
"""End-to-end CPU-vs-MUSA test for the Slice operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_slice_2d():
    data = np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32)
    starts = np.array([0, 4], dtype=np.int64)
    ends = np.array([24, 16], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes, "steps": steps},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_slice_with_step():
    data = np.random.default_rng(1).standard_normal((32,)).astype(np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([32], dtype=np.int64)
    axes = np.array([0], dtype=np.int64)
    steps = np.array([2], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes, "steps": steps},
        outputs=[("Y", TensorProto.FLOAT)],
    )


def test_slice_negative_axes_and_indices_int32_metadata():
    data = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5)
    starts = np.array([-4], dtype=np.int32)
    ends = np.array([-1], dtype=np.int32)
    axes = np.array([-1], dtype=np.int32)
    steps = np.array([2], dtype=np.int32)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes, "steps": steps},
        outputs=[("Y", TensorProto.INT32)],
    )


def test_slice_bool_default_axes():
    data = np.array([[[True, False], [False, True], [True, True]]], dtype=np.bool_)
    starts = np.array([0, 1, 0], dtype=np.int64)
    ends = np.array([1, 3, 2], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends},
        outputs=[("Y", TensorProto.BOOL)],
    )


def test_slice_empty_result():
    data = np.arange(4 * 5, dtype=np.int64).reshape(4, 5)
    starts = np.array([3], dtype=np.int64)
    ends = np.array([1], dtype=np.int64)
    axes = np.array([0], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes},
        outputs=[("Y", TensorProto.INT64)],
    )


def test_slice_float16_3d_step():
    data = np.random.default_rng(2).standard_normal((2, 4, 6)).astype(np.float16)
    starts = np.array([0, 1], dtype=np.int64)
    ends = np.array([2, 6], dtype=np.int64)
    axes = np.array([0, 2], dtype=np.int64)
    steps = np.array([1, 2], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes, "steps": steps},
        outputs=[("Y", TensorProto.FLOAT16)],
    )


def test_slice_uint16_axis1():
    data = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
    starts = np.array([1], dtype=np.int64)
    ends = np.array([5], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    run_and_compare(
        "Slice",
        inputs={"data": data, "starts": starts, "ends": ends, "axes": axes},
        outputs=[("Y", TensorProto.UINT16)],
    )
