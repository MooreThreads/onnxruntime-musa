#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"
#include "plugin_ep_utils.h"

bool IsSliceSumConcatFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSliceSumConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
