#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsSparseIdToMaskFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSparseIdToMaskFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
