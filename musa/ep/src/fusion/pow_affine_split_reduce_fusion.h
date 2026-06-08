#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsPowAffineSplitReduceFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreatePowAffineSplitReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
