// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "fusion/fusion_node_compute.h"

struct ParallelMatMulConcatScratch;

struct ParallelMatMulConcatFusionCompute : FusionNodeCompute {
  ParallelMatMulConcatFusionCompute(size_t input_index,
                                    std::vector<size_t> weight_indices,
                                    int64_t concat_axis,
                                    bool weights_are_initializers);
  ~ParallelMatMulConcatFusionCompute() override;

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t input_index;
  std::vector<size_t> weight_indices;
  int64_t concat_axis;
  bool weights_are_initializers;
  mutable std::unique_ptr<ParallelMatMulConcatScratch> scratch;
  mutable std::mutex scratch_mutex;
};

bool IsParallelMatMulConcatFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateParallelMatMulConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
