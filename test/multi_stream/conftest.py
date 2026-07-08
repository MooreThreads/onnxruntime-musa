# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the MIT License.
"""pytest configuration for MUSA multi-stream tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from op_test_utils import musa_available  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if musa_available():
        return
    skip_musa = pytest.mark.skip(
        reason="No MUSA device available; skipping MUSA multi-stream tests"
    )
    for item in items:
        item.add_marker(skip_musa)
