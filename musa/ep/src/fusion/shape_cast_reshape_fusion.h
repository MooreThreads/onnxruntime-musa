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

struct ShapeCastReshapeTerm {
  enum class Kind {
    kConstant,
    kInputScalar,
  };

  Kind kind = Kind::kConstant;
  int64_t value = 0;
  size_t input_index = 0;
};

struct ShapeCastReshapePlan {
  size_t data_input_index = 0;
  size_t output_index = 0;
  std::vector<ShapeCastReshapeTerm> requested_shape_terms;
  int64_t allowzero = 0;
};

struct ShapeCastReshapeFusionCompute : FusionNodeCompute {
  explicit ShapeCastReshapeFusionCompute(
      std::vector<ShapeCastReshapePlan> outputs);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  std::vector<ShapeCastReshapePlan> outputs;
};

bool IsShapeCastReshapeFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateShapeCastReshapeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
