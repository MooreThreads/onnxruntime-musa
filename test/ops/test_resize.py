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
"""End-to-end CPU-vs-MUSA tests for the supported Resize route."""

import numpy as np

from op_test_utils import TensorProto, run_and_compare


def test_resize_opset17_float_nchw_nearest_asymmetric_floor_scales():
    x = np.arange(1 * 3 * 5 * 7, dtype=np.float32).reshape(1, 3, 5, 7)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    run_and_compare(
        "Resize",
        inputs={"X": x, "roi": np.empty((0,), dtype=np.float32), "scales": scales},
        outputs=[("Y", TensorProto.FLOAT)],
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
        opset=17,
        rtol=0,
        atol=0,
    )
