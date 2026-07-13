// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "fusion/fusion_node_compute.h"
#include "kernels/tensor/slice_concat_impl.h"

struct SliceConcatInput {
  size_t input_index;
  int64_t input_cols;
  int64_t start_col;
  int64_t width;
  int64_t dst_offset;
  bool zero_fill = false;
};

struct SliceConcatFusionCompute : FusionNodeCompute {
  SliceConcatFusionCompute(int64_t output_cols,
                           std::vector<SliceConcatInput> inputs);
  ~SliceConcatFusionCompute() override;

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override;

  int64_t output_cols;
  std::vector<SliceConcatInput> inputs;
  size_t input_slot_count = 0;
  int64_t equal_width = 0;
  int32_t equal_width_shift = -1;
};

bool IsSliceConcatFusionGraph(Ort::ConstGraph graph);
std::unique_ptr<FusionNodeCompute> CreateSliceConcatFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
