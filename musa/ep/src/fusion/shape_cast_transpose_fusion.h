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

struct ShapeCastTransposeTerm {
  bool from_data_dim0 = false;
  std::vector<int64_t> values;
};

struct ShapeCastTransposePlan {
  size_t data_input_index = 0;
  size_t output_index = 0;
  std::vector<ShapeCastTransposeTerm> shape_terms;
  std::vector<int64_t> perm;
  int64_t allowzero = 0;
};

struct ShapeCastTransposeFusionCompute : FusionNodeCompute {
  explicit ShapeCastTransposeFusionCompute(
      std::vector<ShapeCastTransposePlan> outputs);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  std::vector<ShapeCastTransposePlan> outputs;
};

bool IsShapeCastTransposeFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateShapeCastTransposeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
