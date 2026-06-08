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

struct ShapeReshapeTerm {
  bool from_shape_dim = false;
  int64_t dim_index = 0;
  int64_t dim_count = 1;
  std::vector<int64_t> values;
};

struct ShapeReshapeOutputPlan {
  enum class Kind { kReshape, kTile, kTileReshape };

  Kind kind = Kind::kReshape;
  size_t data_input_index = 0;
  size_t output_index = 0;
  std::vector<ShapeReshapeTerm> terms;
  std::vector<ShapeReshapeTerm> tile_terms;
  int64_t allowzero = 0;
};

struct ShapeReshapeFusionCompute : FusionNodeCompute {
  ShapeReshapeFusionCompute(size_t shape_source_input_index,
                            std::vector<ShapeReshapeOutputPlan> outputs);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t shape_source_input_index;
  std::vector<ShapeReshapeOutputPlan> outputs;
};

bool IsShapeReshapeFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateShapeReshapeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
