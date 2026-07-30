// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#include "fusion/strided_view_fusion.h"

#include <stdexcept>

#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/transpose_impl.h"

OrtStatus* StridedViewFusionCompute::Compute(
    OrtKernelContext* kernel_context) const {
  try {
    Ort::KernelContext ctx(kernel_context);
    Ort::ConstValue input = ctx.GetInput(input_index);
    auto info = input.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 3 || shape[0] < 0 || shape[1] < 0 || shape[2] < 0 ||
        segment_count <= 0 || shape[0] % segment_count != 0)
      return Ort::GetApi().CreateStatus(
          ORT_INVALID_ARGUMENT,
          "StridedView requires float [segments*sequence, batch, width] input");
    const int64_t sequence = shape[0] / segment_count;
    Ort::UnownedValue output =
        ctx.GetOutput(0, {sequence, shape[1], shape[2] * segment_count});
    if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
        !IsGpuMemory(output.GetTensorMemoryInfo()))
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "StridedView requires MUSA input/output");
    // ORT tensors have contiguous-layout metadata only, so this logical
    // strided view cannot alias input storage.  Emit the final layout in one
    // kernel directly into the output: no temporary tensor and no D2D memcpy.
    MusaTransposeParams params{};
    params.rank = 4;
    params.total_elements = NumElements(shape);
    const auto strides = Strides({segment_count, sequence, shape[1], shape[2]});
    for (int i = 0; i < 4; ++i) params.input_strides[i] = strides[i];
    params.output_dims[0] = sequence;
    params.output_dims[1] = shape[1];
    params.output_dims[2] = segment_count;
    params.output_dims[3] = shape[2];
    params.perm[0] = 1;
    params.perm[1] = 2;
    params.perm[2] = 0;
    params.perm[3] = 3;
    return LaunchStatus(LaunchMusaTransposeKernel(
        input.GetTensorRawData(), output.GetTensorMutableRawData(),
        sizeof(float), params, GetComputeStream(ctx)));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

bool IsStridedViewFusionGraph(Ort::ConstGraph graph) {
  int concat = 0, slice = 0;
  for (Ort::ConstNode node : graph.GetNodes()) {
    if (node.GetOperatorType() == "Concat")
      ++concat;
    else if (node.GetOperatorType() == "Slice")
      ++slice;
    else
      return false;
  }
  return concat == 1 && slice >= 2;
}

std::unique_ptr<FusionNodeCompute> CreateStridedViewFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  int64_t segments = 0;
  std::string source_name;
  for (Ort::ConstNode node : graph.GetNodes())
    if (node.GetOperatorType() == "Slice") {
      ++segments;
      source_name = node.GetInputs()[0].GetName();
    }
  if (segments < 2) throw std::runtime_error("invalid StridedView graph");
  size_t input_index = 0;
  auto inputs = fused_node.GetInputs();
  for (; input_index < inputs.size(); ++input_index)
    if (inputs[input_index].GetName() == source_name) break;
  if (input_index == inputs.size())
    throw std::runtime_error("StridedView input missing");
  return std::make_unique<StridedViewFusionCompute>(segments, input_index);
}
