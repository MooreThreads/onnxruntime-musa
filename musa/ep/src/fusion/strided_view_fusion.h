// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

struct StridedViewFusionCompute : FusionNodeCompute {
  StridedViewFusionCompute(int64_t segment_count, size_t input_index)
      : segment_count(segment_count), input_index(input_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  int64_t segment_count;
  size_t input_index;
};

bool IsStridedViewFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateStridedViewFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
