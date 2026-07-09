// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsConcatReshapeFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateConcatReshapeFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
