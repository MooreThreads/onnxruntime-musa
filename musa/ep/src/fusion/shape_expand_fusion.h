// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "fusion/fusion_node_compute.h"

struct ShapeExpandTerm {
  bool from_shape_dim = false;
  int64_t dim_index = 0;
  int64_t dim_count = 1;
  std::vector<int64_t> values;
};

struct ShapeExpandOutputPlan {
  enum class Kind { Expand, ConstantOfShape };

  Kind kind = Kind::Expand;
  size_t data_input_index = 0;
  size_t output_index = 0;
  std::vector<ShapeExpandTerm> terms;
  ONNXTensorElementDataType value_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  uint64_t value_bits = 0;
  size_t value_size = sizeof(float);
  bool has_scalar_fill_value = false;
  ONNXTensorElementDataType scalar_fill_value_type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  uint64_t scalar_fill_value_bits = 0;
  size_t scalar_fill_value_size = 0;
};

struct ShapeExpandFusionCompute : FusionNodeCompute {
  ShapeExpandFusionCompute(size_t shape_source_input_index,
                           std::vector<ShapeExpandOutputPlan> outputs);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t shape_source_input_index;
  std::vector<ShapeExpandOutputPlan> outputs;
};

bool IsShapeExpandFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateShapeExpandFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
