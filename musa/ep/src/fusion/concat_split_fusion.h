// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

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
