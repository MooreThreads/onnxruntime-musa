// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

namespace musa_ep {

bool IsGenerateMTGRCustomMaskFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateGenerateMTGRCustomMaskFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);

}  // namespace musa_ep
