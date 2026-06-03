// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsLinearFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateLinearFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
