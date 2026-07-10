// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "fusion/fusion_node_compute.h"

bool IsMhtaScaledDotProductAttentionFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateMhtaScaledDotProductAttentionFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
