// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "fusion/fusion_node_compute.h"

struct SplitUnsqueezeConcatFusionCompute : FusionNodeCompute {
  SplitUnsqueezeConcatFusionCompute(size_t input_index, int64_t sequence,
                                    int64_t part_count, int64_t part_width,
                                    bool transpose, size_t shape_input_index);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t input_index;
  int64_t sequence;
  int64_t part_count;
  int64_t part_width;
  bool transpose;
  size_t shape_input_index;
};

bool IsSplitUnsqueezeConcatFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSplitUnsqueezeConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
