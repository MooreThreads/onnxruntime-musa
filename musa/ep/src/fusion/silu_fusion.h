// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <memory>

#include "fusion/fusion_node_compute.h"

namespace musa_ep {

struct SiluFusionCompute final : FusionNodeCompute {
  explicit SiluFusionCompute(size_t input_index) : input_index(input_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t input_index;
};

bool IsSiluFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSiluFusion(Ort::ConstGraph graph,
                                                    Ort::ConstNode fused_node);

}  // namespace musa_ep
