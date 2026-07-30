#pragma once

#include <memory>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "fusion/fusion_node_compute.h"

bool IsReducedMhaFlashFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateReducedMhaFlashFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
