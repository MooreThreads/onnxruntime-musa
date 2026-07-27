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

class MusaEp;

// Runtime implementation for one ORT fused node.
//
// GetCapability selects a supported subgraph pattern and ORT replaces that
// subgraph with one fused node assigned to this EP. CompileImpl creates one
// FusionNodeCompute object for each fused node name; FusionNodeComputeInfo then
// looks it up at runtime and forwards ORT's kernel context to Compute().
struct FusionNodeCompute {
  virtual ~FusionNodeCompute() = default;
  virtual OrtStatus* Compute(OrtKernelContext* kernel_context) const = 0;
};

OrtNodeComputeInfo* CreateFusionNodeComputeInfo(MusaEp& ep);
void ReleaseFusionNodeComputeInfo(OrtNodeComputeInfo* node_compute_info);
