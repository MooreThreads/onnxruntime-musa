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

struct ShapeCastSplitTerm {
  bool from_data_dim0 = false;
  std::vector<int64_t> values;
};

struct ShapeCastSplitOutputPlan {
  size_t output_index = 0;
  int64_t axis_offset = 0;
  int64_t axis_width = 0;
};

struct ShapeCastSplitFusionCompute : FusionNodeCompute {
  ShapeCastSplitFusionCompute(size_t data_input_index,
                              std::vector<ShapeCastSplitTerm> shape_terms,
                              int64_t split_axis,
                              std::vector<ShapeCastSplitOutputPlan> outputs,
                              int64_t allowzero);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t data_input_index = 0;
  std::vector<ShapeCastSplitTerm> shape_terms;
  int64_t split_axis = 0;
  std::vector<ShapeCastSplitOutputPlan> outputs;
  int64_t allowzero = 0;
};

bool IsShapeCastSplitFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateShapeCastSplitFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
