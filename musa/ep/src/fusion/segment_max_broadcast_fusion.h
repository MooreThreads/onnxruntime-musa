#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsSegmentMaxBroadcastFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSegmentMaxBroadcastFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
