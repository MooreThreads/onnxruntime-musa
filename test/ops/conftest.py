# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
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
