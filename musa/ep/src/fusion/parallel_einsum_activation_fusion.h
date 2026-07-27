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
#include <memory>
#include <mutex>
#include <vector>

#include "fusion/fusion_node_compute.h"

struct ParallelEinsumActivationDeviceConstants;

struct ParallelEinsumActivationBranchInputs {
  size_t w1_index;
  size_t w2_index;
  size_t w3_index;
};

struct ParallelEinsumActivationFusionCompute : FusionNodeCompute {
  ParallelEinsumActivationFusionCompute(
      size_t mlp_input_index, size_t gate_input_index, size_t bias_index,
      std::vector<ParallelEinsumActivationBranchInputs> branches,
      bool constants_are_initializers);
  ~ParallelEinsumActivationFusionCompute() override;

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t mlp_input_index;
  size_t gate_input_index;
  size_t bias_index;
  std::vector<ParallelEinsumActivationBranchInputs> branches;
  bool constants_are_initializers;
  mutable std::unique_ptr<ParallelEinsumActivationDeviceConstants> constants;
  mutable std::mutex constants_mutex;
};

bool IsParallelEinsumActivationFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateParallelEinsumActivationFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
