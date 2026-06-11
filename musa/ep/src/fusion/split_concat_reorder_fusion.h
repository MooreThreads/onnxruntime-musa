// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "fusion/fusion_node_compute.h"

struct SplitConcatReorderFusionCompute : FusionNodeCompute {
  SplitConcatReorderFusionCompute(size_t input_index, int64_t sequence,
                                  int64_t part_count, int64_t part_width);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t input_index;
  int64_t sequence;
  int64_t part_count;
  int64_t part_width;
};

bool IsSplitConcatReorderFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSplitConcatReorderFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
