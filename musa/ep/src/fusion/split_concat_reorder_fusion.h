// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
