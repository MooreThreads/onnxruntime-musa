// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

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
