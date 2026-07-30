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
"""End-to-end CPU-vs-MUSA test for the Split operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_split_even_axis0():
    x = np.random.default_rng(0).standard_normal((32, 16)).astype(np.float32)
    split = np.array([16, 16], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        attrs={"axis": 0},
    )


def test_split_uneven_axis1():
    x = np.random.default_rng(1).standard_normal((32, 48)).astype(np.float32)
    split = np.array([16, 32], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT), ("Y1", TensorProto.FLOAT)],
        attrs={"axis": 1},
    )


def test_split_negative_axis_int32():
    x = np.arange(2 * 3 * 5, dtype=np.int32).reshape(2, 3, 5)
    split = np.array([2, 3], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.INT32), ("Y1", TensorProto.INT32)],
        attrs={"axis": -1},
    )


def test_split_bool_three_outputs():
    x = np.array([[True, False, True, False, True, False]], dtype=np.bool_)
    split = np.array([1, 2, 3], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[
            ("Y0", TensorProto.BOOL),
            ("Y1", TensorProto.BOOL),
            ("Y2", TensorProto.BOOL),
        ],
        attrs={"axis": 1},
    )


def test_split_float16_axis0():
    x = np.random.default_rng(2).standard_normal((6, 4)).astype(np.float16)
    split = np.array([2, 4], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[("Y0", TensorProto.FLOAT16), ("Y1", TensorProto.FLOAT16)],
        attrs={"axis": 0},
    )


def test_split_uint8_three_outputs():
    x = np.arange(2 * 6, dtype=np.uint8).reshape(2, 6)
    split = np.array([1, 2, 3], dtype=np.int64)
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=[
            ("Y0", TensorProto.UINT8),
            ("Y1", TensorProto.UINT8),
            ("Y2", TensorProto.UINT8),
        ],
        attrs={"axis": 1},
    )


def test_split_many_small_outputs_axis1():
    x = np.random.default_rng(3).standard_normal((8, 80)).astype(np.float32)
    split = np.array([1, 3] * 20, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.FLOAT) for i in range(split.size)]
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=outputs,
        attrs={"axis": 1},
    )


def test_split_thirteen_rank3_outputs_axis1():
    x = np.random.default_rng(4).standard_normal((4, 13, 768)).astype(np.float32)
    split = np.ones(13, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.FLOAT) for i in range(split.size)]
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=outputs,
        attrs={"axis": 1},
    )


def test_split_equal_three_outputs_fp32():
    x = np.random.default_rng(5).standard_normal((5, 3072)).astype(np.float32)
    split = np.full(3, 1024, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.FLOAT) for i in range(split.size)]
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=outputs,
        attrs={"axis": 1},
    )


def test_split_equal_six_outputs_fp16():
    x = np.random.default_rng(6).standard_normal((2, 30, 96)).astype(np.float16)
    split = np.full(6, 16, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.FLOAT16) for i in range(split.size)]
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=outputs,
        attrs={"axis": 2},
    )


def test_split_equal_four_outputs_int64():
    x = np.arange(4, dtype=np.int64)
    split = np.ones(4, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.INT64) for i in range(split.size)]
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=outputs,
        attrs={"axis": 0},
    )


def test_split_equal_three_outputs_bfloat16():
    x = float32_to_bfloat16_bits(
        np.random.default_rng(7).standard_normal((2, 6, 8)).astype(np.float32)
    )
    split = np.full(3, 2, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.BFLOAT16) for i in range(split.size)]
    model = build_model_with_input_types(
        "Split",
        inputs={"X": x, "split": split},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=outputs,
        attrs={"axis": 1},
    )
    actual = run_with_iobinding(
        model,
        {"X": x, "split": split},
        {"X": TensorProto.BFLOAT16},
        [(name, TensorProto.BFLOAT16, (2, 2, 8)) for name, _ in outputs],
        use_musa=True,
    )
    for output, expected in zip(actual, np.split(x, 3, axis=1)):
        np.testing.assert_array_equal(output, expected)


def test_split_equal_thirty_three_outputs_fallback():
    x = np.random.default_rng(8).standard_normal((2, 33, 4)).astype(np.float32)
    split = np.ones(33, dtype=np.int64)
    outputs = [(f"Y{i}", TensorProto.FLOAT) for i in range(split.size)]
    run_and_compare(
        "Split",
        inputs={"X": x, "split": split},
        outputs=outputs,
        attrs={"axis": 1},
    )
