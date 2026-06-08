# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Pytest defaults for MUSA correctness tests."""

import os

# Correctness tests compare MUSA outputs against CPU FP32 references with tight
# tolerances. Keep TF32 off by default for pytest, while still allowing explicit
# env overrides for targeted TF32 coverage.
os.environ.setdefault("MUSA_ENABLE_MUDNN_TF32", "0")
os.environ.setdefault("MUSA_ENABLE_MUBLAS_TF32", "0")
