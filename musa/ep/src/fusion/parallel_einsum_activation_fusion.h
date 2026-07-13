// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

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
