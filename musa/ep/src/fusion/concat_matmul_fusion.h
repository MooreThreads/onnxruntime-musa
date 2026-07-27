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
