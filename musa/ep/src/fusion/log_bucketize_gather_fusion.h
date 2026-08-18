#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

bool IsLogBucketizeGatherFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateLogBucketizeGatherFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
