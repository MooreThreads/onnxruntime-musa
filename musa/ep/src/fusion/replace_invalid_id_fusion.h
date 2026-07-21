#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsReplaceInvalidIdFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateReplaceInvalidIdFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
