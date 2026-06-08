// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstddef>
#include <memory>
#include <vector>

#include "fusion/fusion_node_compute.h"

struct TileMaskSelectPlan {
  size_t true_input_index = 0;
  size_t false_input_index = 0;
  size_t output_index = 0;
};

struct TileMaskSelectFusionCompute : FusionNodeCompute {
  TileMaskSelectFusionCompute(size_t mask_input_index,
                              std::vector<TileMaskSelectPlan> outputs);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t mask_input_index;
  std::vector<TileMaskSelectPlan> outputs;
};

bool IsTileMaskSelectFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateTileMaskSelectFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
