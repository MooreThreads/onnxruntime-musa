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

#include <span>
#include <unordered_set>

namespace musa_ep {

OrtStatus* GetCpuPreferredNodes(
    const OrtGraph& ort_graph, OrtEpGraphSupportInfo& graph_support_info,
    const OrtEpApi& ep_api, std::span<const OrtNode* const> tentative_nodes,
    std::unordered_set<const OrtNode*>& cpu_preferred_nodes) noexcept;

}  // namespace musa_ep
