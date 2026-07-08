#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsMathConcatLogFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateMathConcatLogFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
