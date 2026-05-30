// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "common/op_kernel_common.h"

enum class ReduceMode { kProd, kSum, kMean };

// Shared host-side reduction used by ReduceProd / ReduceSum / ReduceMean.
inline OrtStatus* ReduceCompute(Ort::KernelContext& ctx,
                                std::vector<int64_t> axes, bool keepdims,
                                ReduceMode mode) {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  if (ctx.GetInputCount() > 1) axes = ReadIntTensor(ctx, 1);
  auto axes_set = AxesSet(axes, shape0.size());

  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < shape0.size(); ++i) {
    if (axes_set.count(static_cast<int64_t>(i))) {
      if (keepdims) out_shape.push_back(1);
    } else {
      out_shape.push_back(shape0[i]);
    }
  }
  if (out_shape.empty()) out_shape.push_back(1);
  auto out_strides = Strides(out_shape);

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    std::vector<int64_t> x = ReadTyped<int64_t>(input0);
    std::vector<int64_t> out(static_cast<size_t>(NumElements(out_shape)),
                             mode == ReduceMode::kProd ? 1 : 0);
    for (int64_t i = 0; i < NumElements(shape0); ++i) {
      auto ic = Coordinates(i, shape0);
      std::vector<int64_t> oc;
      for (size_t d = 0; d < shape0.size(); ++d) {
        if (axes_set.count(static_cast<int64_t>(d))) {
          if (keepdims) oc.push_back(0);
        } else {
          oc.push_back(ic[d]);
        }
      }
      if (oc.empty()) oc.push_back(0);
      int64_t oo = Offset(oc, out_strides);
      if (mode == ReduceMode::kProd)
        out[static_cast<size_t>(oo)] *= x[static_cast<size_t>(i)];
      else
        out[static_cast<size_t>(oo)] += x[static_cast<size_t>(i)];
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return WriteTyped<int64_t>(y, out);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> x = ReadTyped<float>(input0);
    std::vector<float> out(static_cast<size_t>(NumElements(out_shape)),
                           mode == ReduceMode::kProd ? 1.0f : 0.0f);
    std::vector<int64_t> counts(out.size(), 0);
    for (int64_t i = 0; i < NumElements(shape0); ++i) {
      auto ic = Coordinates(i, shape0);
      std::vector<int64_t> oc;
      for (size_t d = 0; d < shape0.size(); ++d) {
        if (axes_set.count(static_cast<int64_t>(d))) {
          if (keepdims) oc.push_back(0);
        } else {
          oc.push_back(ic[d]);
        }
      }
      if (oc.empty()) oc.push_back(0);
      int64_t oo = Offset(oc, out_strides);
      if (mode == ReduceMode::kProd)
        out[static_cast<size_t>(oo)] *= x[static_cast<size_t>(i)];
      else
        out[static_cast<size_t>(oo)] += x[static_cast<size_t>(i)];
      counts[static_cast<size_t>(oo)]++;
    }
    if (mode == ReduceMode::kMean) {
      for (size_t i = 0; i < out.size(); ++i)
        out[i] /= static_cast<float>(counts[i]);
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return WriteTyped<float>(y, out);
  }
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "unsupported reduce dtype");
}
