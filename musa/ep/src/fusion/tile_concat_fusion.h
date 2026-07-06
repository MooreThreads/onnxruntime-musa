// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsTileConcatFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateTileConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
