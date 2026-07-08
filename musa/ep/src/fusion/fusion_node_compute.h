// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

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
