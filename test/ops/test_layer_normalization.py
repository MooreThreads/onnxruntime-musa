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
"""End-to-end CPU-vs-MUSA test for the LayerNormalization operator."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_layer_normalization_float_axis_minus1():
    x = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
    scale = np.linspace(0.5, 1.5, 4, dtype=np.float32)
    bias = np.linspace(-0.2, 0.2, 4, dtype=np.float32)
    run_and_compare(
        "LayerNormalization",
        inputs={"X": x, "Scale": scale, "B": bias},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={"axis": -1, "epsilon": 1e-5},
        opset=17,
        rtol=1e-4,
        atol=1e-4,
    )
