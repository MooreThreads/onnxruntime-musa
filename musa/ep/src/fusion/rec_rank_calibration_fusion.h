#pragma once

#include <memory>

#include "fusion/fusion_node_compute.h"

namespace musa_ep {

bool IsRecRankCalibrationFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateRecRankCalibrationFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);

}  // namespace musa_ep
