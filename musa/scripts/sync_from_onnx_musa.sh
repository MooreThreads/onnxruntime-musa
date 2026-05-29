#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-/home/workspace/onnx_musa}"
DST_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Source: ${SRC_ROOT}"
echo "Target: ${DST_ROOT}"
echo
echo "This script documents the migration mapping. It intentionally does not overwrite files yet."
echo "runtime -> musa/ep/src/runtime"
echo "core/providers/musa/* kernels -> musa/ep/src/kernels"
echo "contrib_ops/musa fused kernels -> musa/ep/src/kernels/fused"
echo "core/optimizer/musa_operator_fusion.* -> musa/ep/src/graph/fusion_patterns.*"
