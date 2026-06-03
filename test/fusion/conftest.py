# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""pytest configuration for graph fusion tests.

Fusion tests reuse the MUSA plugin registration and CPU-vs-MUSA helpers from
test/ops/op_test_utils.py, but live in a separate suite because they validate
multi-node graph patterns instead of single operator kernels.
"""

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
    skip_musa = pytest.mark.skip(reason="No MUSA device available; skipping CPU-vs-MUSA tests")
    for item in items:
        item.add_marker(skip_musa)
