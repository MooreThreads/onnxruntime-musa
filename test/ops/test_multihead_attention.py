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
"""End-to-end tests for com.microsoft::MultiHeadAttention on MUSA."""

import numpy as np
import pytest

from op_test_utils import TensorProto, build_model, run, run_and_compare


def test_ms_multihead_attention_fp32_rank3_mask():
    rng = np.random.default_rng(20260730)
    batch, sequence, hidden, heads = 1, 7, 16, 4
    query = (rng.standard_normal((batch, sequence, hidden)) * 0.2).astype(np.float32)
    key = (rng.standard_normal((batch, sequence, hidden)) * 0.2).astype(np.float32)
    value = (rng.standard_normal((batch, sequence, hidden)) * 0.2).astype(np.float32)
    bias = (rng.standard_normal(3 * hidden) * 0.05).astype(np.float32)
    mask = np.ones((batch, sequence, sequence), dtype=np.int32)
    mask[:, :, -2:] = 0

    run_and_compare(
        "MultiHeadAttention",
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "bias": bias,
            "key_padding_mask": mask,
        },
        outputs=[("output", TensorProto.FLOAT)],
        attrs={"num_heads": heads},
        domain="com.microsoft",
        rtol=3e-4,
        atol=3e-4,
    )


def test_ms_multihead_attention_fp32_mask_filter_value_all_masked_row():
    rng = np.random.default_rng(197)
    batch, sequence, hidden, heads = 1, 5, 8, 2
    query = (rng.standard_normal((batch, sequence, hidden)) * 0.3).astype(np.float32)
    key = (rng.standard_normal((batch, sequence, hidden)) * 0.3).astype(np.float32)
    value = (rng.standard_normal((batch, sequence, hidden)) * 0.3).astype(np.float32)
    bias = (rng.standard_normal(3 * hidden) * 0.1).astype(np.float32)
    mask = np.ones((batch, sequence, sequence), dtype=np.int32)
    mask[:, 2, :] = 0
    mask[:, :, -1] = 0
    mask[:, 1, 0] = -1

    run_and_compare(
        "MultiHeadAttention",
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "bias": bias,
            "key_padding_mask": mask,
        },
        outputs=[("output", TensorProto.FLOAT)],
        attrs={
            "num_heads": heads,
            "scale": 0.37,
            "mask_filter_value": -2.0,
        },
        domain="com.microsoft",
        rtol=3e-4,
        atol=3e-4,
    )


def test_ms_multihead_attention_fp32_without_mask():
    rng = np.random.default_rng(311)
    batch, sequence, hidden, heads = 2, 4, 12, 3
    query = (rng.standard_normal((batch, sequence, hidden)) * 0.2).astype(np.float32)
    key = (rng.standard_normal((batch, sequence, hidden)) * 0.2).astype(np.float32)
    value = (rng.standard_normal((batch, sequence, hidden)) * 0.2).astype(np.float32)
    bias = (rng.standard_normal(3 * hidden) * 0.05).astype(np.float32)

    run_and_compare(
        "MultiHeadAttention",
        inputs={"query": query, "key": key, "value": value, "bias": bias},
        outputs=[("output", TensorProto.FLOAT)],
        attrs={"num_heads": heads},
        domain="com.microsoft",
        rtol=3e-4,
        atol=3e-4,
    )


def test_ms_multihead_attention_rejects_unidirectional():
    query = np.zeros((1, 2, 4), dtype=np.float32)
    key = np.zeros_like(query)
    value = np.zeros_like(query)
    bias = np.zeros(12, dtype=np.float32)
    inputs = {"query": query, "key": key, "value": value, "bias": bias}
    model = build_model(
        "MultiHeadAttention",
        inputs=inputs,
        outputs=[("output", TensorProto.FLOAT)],
        attrs={"num_heads": 2, "unidirectional": 1},
        domain="com.microsoft",
    )

    with pytest.raises(Exception, match="unidirectional"):
        run(model, inputs, use_musa=True)
