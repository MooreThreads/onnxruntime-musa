#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsCenteredReduceFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateCenteredReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
