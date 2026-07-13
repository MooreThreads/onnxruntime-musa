// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fusion/fusion_node_compute.h"

struct ConcatMatMulFusionCompute : FusionNodeCompute {
  ConcatMatMulFusionCompute(int64_t axis, int64_t concat_input_idx,
                            std::vector<size_t> concat_input_indices,
                            size_t other_input_index,
                            std::vector<std::string> fused_input_names);
  ~ConcatMatMulFusionCompute() override;

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  int64_t axis;
  int64_t concat_input_idx;
  std::vector<size_t> concat_input_indices;
  size_t other_input_index;
  std::vector<std::string> fused_input_names;
};

std::unique_ptr<FusionNodeCompute> CreateConcatMatMulFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
