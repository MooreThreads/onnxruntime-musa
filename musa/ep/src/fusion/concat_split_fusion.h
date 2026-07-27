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

#include <memory>
#include <vector>

#include "fusion/fusion_node_compute.h"
#include "kernels/tensor/concat_split_impl.h"

struct ConcatSplitOutput {
  int64_t width;
};

struct ConcatSplitSegmentSpec {
  size_t source_input_index;
  size_t output_index;
  int64_t source_offset;
  int64_t width;
  int64_t dst_offset;
};

struct ConcatSplitSumTermSpec {
  size_t source_input_index;
  int64_t source_offset;
};

struct ConcatSplitSumSpec {
  size_t output_index;
  int64_t width;
  std::vector<ConcatSplitSumTermSpec> terms;
};

struct ConcatSplitFusionCompute : FusionNodeCompute {
  ConcatSplitFusionCompute(std::vector<ConcatSplitOutput> outputs,
                           std::vector<ConcatSplitSegmentSpec> segments,
                           std::vector<ConcatSplitSumSpec> sums);
  ~ConcatSplitFusionCompute() override;

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  std::vector<ConcatSplitOutput> outputs;
  std::vector<ConcatSplitSegmentSpec> segments;
  std::vector<ConcatSplitSumSpec> sums;
};

bool IsConcatSplitFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateConcatSplitFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
