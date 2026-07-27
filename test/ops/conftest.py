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
"""pytest configuration shared by all op tests.

Adds this directory to sys.path so `import op_test_utils` works, and registers
a global skip when no MUSA device is available so the suite can still be
collected/run on CPU-only machines.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402

from op_test_utils import musa_available  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if musa_available():
        return
    skip_musa = pytest.mark.skip(reason="No MUSA device available; skipping CPU-vs-MUSA tests")
    for item in items:
        item.add_marker(skip_musa)
