// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <memory>
#include <string>
#include <vector>

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_node_compute.h"

bool IsTileConcatFusionGraph(Ort::ConstGraph graph);

// Identifies one Tile+Concat fusion across GetCapability() and Compile(): both
// sides see the fusion boundary (its input and output value names) but not each
// other's nodes. Names are sorted and de-duplicated so the order ORT chooses
// for the fused node's inputs does not matter.
std::string TileConcatFusionKey(const std::vector<std::string>& input_names,
                                const std::vector<std::string>& output_names);

// `shape_expr_hints` carries the repeats expressions the matcher proved on the
// whole graph; Compile() cannot see that chain any more.
std::unique_ptr<FusionNodeCompute> CreateTileConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node,
    const musa_ep::FusionShapeExprHints* shape_expr_hints = nullptr);
