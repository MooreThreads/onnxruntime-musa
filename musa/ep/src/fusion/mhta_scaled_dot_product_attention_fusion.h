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

#include <memory>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "fusion/fusion_matcher.h"
#include "fusion/fusion_node_compute.h"

bool IsMhtaScaledDotProductAttentionFusionGraph(Ort::ConstGraph graph);
// `mask_end_hints` carries the Slice-end provenance the matcher proved on the
// whole graph; Compile() cannot see the end's producer chain any more.
std::unique_ptr<FusionNodeCompute> CreateMhtaScaledDotProductAttentionFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node,
    const musa_ep::FusionMaskEndHints* mask_end_hints = nullptr);
