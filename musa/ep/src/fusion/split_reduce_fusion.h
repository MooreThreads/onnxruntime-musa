#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsSplitReduceFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSplitReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
