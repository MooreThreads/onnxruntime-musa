// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstddef>
#include <memory>

#include "fusion/fusion_node_compute.h"
#include "kernels/shared_inc/device_kernel_types.h"

struct MaskedGatherReduceFusionCompute : FusionNodeCompute {
  MaskedGatherReduceFusionCompute(size_t mask_input_index,
                                  size_t data_input_index,
                                  size_t output_index,
                                  MusaReduceOp reduce_op);

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  size_t mask_input_index;
  size_t data_input_index;
  size_t output_index;
  MusaReduceOp reduce_op;
};

bool IsMaskedGatherReduceFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateMaskedGatherReduceFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
