// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fusion/fusion_node_compute.h"

namespace musa_ep {

struct SwiGluFusionCompute final : FusionNodeCompute {
  SwiGluFusionCompute(size_t input_index, size_t output_index,
                      ONNXTensorElementDataType elem_type,
                      std::vector<int64_t> weight_shape,
                      std::vector<uint8_t> packed_weights);
  ~SwiGluFusionCompute() override;

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool IsSwiGluFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSwiGluFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);

}  // namespace musa_ep
