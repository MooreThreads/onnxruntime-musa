# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""End-to-end tests for com.microsoft::Attention on MUSA."""

import numpy as np

from op_test_utils import TensorProto, build_model, run


def _attention_reference(x, weights, bias, mask, num_heads, scale, qkv_hidden_sizes):
    batch, sequence, _ = x.shape
    q_hidden, k_hidden, v_hidden = qkv_hidden_sizes
    q_head = q_hidden // num_heads
    k_head = k_hidden // num_heads
    v_head = v_hidden // num_heads
    qkv = x.reshape(batch * sequence, -1) @ weights + bias
    qkv = qkv.reshape(batch, sequence, q_hidden + k_hidden + v_hidden)
    q = qkv[:, :, :q_hidden].reshape(batch, sequence, num_heads, q_head)
    k = qkv[:, :, q_hidden : q_hidden + k_hidden].reshape(
        batch, sequence, num_heads, k_head
    )
    v = qkv[:, :, q_hidden + k_hidden :].reshape(batch, sequence, num_heads, v_head)

    y = np.zeros((batch, sequence, v_hidden), dtype=np.float32)
    for b in range(batch):
        for h in range(num_heads):
            for i in range(sequence):
                scores = np.array(
                    [np.dot(q[b, i, h], k[b, j, h]) * scale for j in range(sequence)],
                    dtype=np.float32,
                )
                row_mask = mask[0 if mask.shape[0] == 1 else b, 0, i]
                scores = np.where(row_mask != 0, scores, -np.inf)
                weights_row = np.exp(scores - np.max(scores))
                weights_row = weights_row / np.sum(weights_row)
                values = weights_row @ v[b, :, h]
                start = h * v_head
                y[b, i, start : start + v_head] = values
    return y


def test_ms_attention_float_4d_int32_mask():
    x = np.array(
        [
            [
                [0.1, -0.2, 0.3, 0.4],
                [0.5, 0.6, -0.7, 0.8],
                [-0.9, 1.0, 1.1, -1.2],
            ]
        ],
        dtype=np.float32,
    )
    weights = (np.arange(4 * 12, dtype=np.float32).reshape(4, 12) - 9.0) / 20.0
    bias = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    mask = np.array(
        [[[[1, 1, 0], [1, 1, 0], [1, 1, 1]]]],
        dtype=np.int32,
    )
    attrs = {"num_heads": 2, "qkv_hidden_sizes": [4, 4, 4], "scale": 0.5}
    model = build_model(
        "Attention",
        inputs={"input": x, "weights": weights, "bias": bias, "mask": mask},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs=attrs,
        domain="com.microsoft",
        opset=17,
    )

    (actual,) = run(
        model,
        {"input": x, "weights": weights, "bias": bias, "mask": mask},
        use_musa=True,
    )
    expected = _attention_reference(x, weights, bias, mask, 2, 0.5, [4, 4, 4])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
