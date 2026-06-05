// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fusion/fusion_node_compute.h"

struct ShapeCastGatherFusionCompute : FusionNodeCompute {
  ShapeCastGatherFusionCompute(size_t data_input_index,
                               size_t indices_input_index,
                               std::vector<int64_t> cached_indices,
                               int64_t output_type);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t data_input_index;
  size_t indices_input_index;
  std::vector<int64_t> cached_indices;
  int64_t output_type;
};

bool IsShapeCastGatherFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateShapeCastGatherFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
