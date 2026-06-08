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

struct ShapeCastConcatTerm {
  bool from_data_dim = false;
  size_t dim_index = 0;
  std::vector<int64_t> constants;
};

struct ShapeCastConcatOutputPlan {
  size_t output_index = 0;
  int64_t output_type = 0;
  bool full_data_shape = false;
  std::vector<ShapeCastConcatTerm> terms;
};

struct ShapeCastConcatFusionCompute : FusionNodeCompute {
  ShapeCastConcatFusionCompute(size_t data_input_index,
                               std::vector<ShapeCastConcatOutputPlan> outputs);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t data_input_index;
  std::vector<ShapeCastConcatOutputPlan> outputs;
};

bool IsShapeCastConcatFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateShapeCastConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
