// Copyright (c) Microsoft Corporation. All rights reserved.
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

struct ConcatMatMulScratch;

struct ConcatMatMulFusionCompute : FusionNodeCompute {
  ConcatMatMulFusionCompute(int64_t axis, int64_t concat_input_idx,
                            std::vector<size_t> concat_input_indices,
                            size_t other_input_index);
  ~ConcatMatMulFusionCompute() override;

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  int64_t axis;
  int64_t concat_input_idx;
  std::vector<size_t> concat_input_indices;
  size_t other_input_index;
  mutable std::unique_ptr<ConcatMatMulScratch> scratch;
  mutable std::mutex scratch_mutex;
};

std::unique_ptr<FusionNodeCompute> CreateConcatMatMulFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
