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
"""End-to-end CPU-vs-MUSA test for the ReduceProd operator."""

import numpy as np

from op_test_utils import (
    TensorProto,
    bfloat16_bits_to_float32,
    build_model_with_input_types,
    float32_to_bfloat16_bits,
    run_and_compare,
    run_with_iobinding,
)


def test_reduce_prod_axis1_keepdims():
    x = np.random.default_rng(0).uniform(0.5, 1.5, (16, 32)).astype(np.float32)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_prod_1d_negative_axis_no_keepdims_scalar():
    x = np.random.default_rng(7).uniform(0.8, 1.2, (32,)).astype(np.float32)
    outputs = run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [-1], "keepdims": 0},
    )
    assert outputs[0].shape == ()


def test_reduce_prod_int64():
    # Use tile of [1, 2] so each row product = 2**16 (no overflow)
    x = np.tile(np.array([1, 2], dtype=np.int64), (16, 16))
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT64)],
        attrs={"axes": [1], "keepdims": 0},
    )


def test_reduce_prod_int32_axis1_keepdims():
    x = np.tile(np.array([1, 2, 3, 1], dtype=np.int32), (8, 1))
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axes": [1], "keepdims": 1},
    )


def test_reduce_prod_int32_negative_axis_keepdims():
    x = np.tile(np.array([1, 2, 3], dtype=np.int32), (2, 4, 1))
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axes": [-1], "keepdims": 1},
    )


def test_reduce_prod_profile_last_axis_256_keepdims():
    x = np.random.default_rng(6).uniform(0.99, 1.01, (32, 18, 256)).astype(np.float32)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [-1], "keepdims": 1},
        rtol=1e-5,
        atol=1e-5,
    )


def test_reduce_prod_float_axis0_no_keepdims_3d():
    x = np.random.default_rng(2).uniform(0.5, 1.25, (2, 3, 4)).astype(np.float32)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axes": [0], "keepdims": 0},
    )


def test_reduce_prod_multi_axis_int32_no_keepdims():
    x = np.ones((2, 3, 4), dtype=np.int32)
    x[:, :, 0] = 2
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.INT32)],
        attrs={"axes": [0, 2], "keepdims": 0},
    )


def test_reduce_prod_float16_axis1_no_keepdims():
    x = np.random.default_rng(3).uniform(0.8, 1.2, (4, 8)).astype(np.float16)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.FLOAT16)],
        attrs={"axes": [1], "keepdims": 0},
        rtol=2e-2,
        atol=2e-2,
    )


def test_reduce_prod_double_axis0_keepdims():
    x = np.random.default_rng(4).uniform(0.8, 1.2, (4, 8)).astype(np.float64)
    run_and_compare(
        "ReduceProd",
        inputs={"X": x},
        outputs=[("Y", TensorProto.DOUBLE)],
        attrs={"axes": [0], "keepdims": 1},
        rtol=1e-6,
        atol=1e-7,
    )


def test_reduce_prod_bfloat16_axis1_no_keepdims():
    x_f32 = np.random.default_rng(5).uniform(0.8, 1.2, (4, 8)).astype(np.float32)
    x = float32_to_bfloat16_bits(x_f32)
    x_bf16_f32 = bfloat16_bits_to_float32(x)
    expected = np.prod(x_bf16_f32, axis=1)
    model = build_model_with_input_types(
        "ReduceProd",
        inputs={"X": x},
        input_types={"X": TensorProto.BFLOAT16},
        outputs=[("Y", TensorProto.BFLOAT16)],
        attrs={"axes": [1], "keepdims": 0},
    )
    (actual,) = run_with_iobinding(
        model,
        {"X": x},
        {"X": TensorProto.BFLOAT16},
        [("Y", TensorProto.BFLOAT16, expected.shape)],
        use_musa=True,
    )
    np.testing.assert_allclose(
        bfloat16_bits_to_float32(actual),
        expected,
        rtol=3e-2,
        atol=3e-2,
    )
