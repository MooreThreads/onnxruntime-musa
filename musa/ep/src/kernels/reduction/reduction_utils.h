// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "shared_inc/op_kernel_common.h"
#include "reduction/reduction_functions.h"

enum class ReduceMode { kProd, kSum, kMean, kSumSquare };

// Shared host-side reduction used by ReduceProd / ReduceSum / ReduceMean /
// ReduceSumSquare.
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
      else if (mode == ReduceMode::kSumSquare)
        out[static_cast<size_t>(oo)] +=
            x[static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
      else
        out[static_cast<size_t>(oo)] += x[static_cast<size_t>(i)];
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return WriteTyped<int64_t>(y, out);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    if (axes_set.size() == 1 &&
        shape0.size() <= kMusaMaxBroadcastRank &&
        IsGpuMemory(input0.GetTensorMemoryInfo())) {
      Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
      if (IsGpuMemory(y.GetTensorMemoryInfo())) {
        const int64_t reduce_axis = *axes_set.begin();
        auto in_strides = Strides(shape0);
        auto output_strides = Strides(out_shape);
        MusaReduceParams params{};
        params.rank = static_cast<int32_t>(shape0.size());
        params.reduce_axis = static_cast<int32_t>(reduce_axis);
        params.output_elements = NumElements(out_shape);
        params.reduce_dim = shape0[static_cast<size_t>(reduce_axis)];
        size_t out_dim = 0;
        for (size_t dim = 0; dim < shape0.size(); ++dim) {
          params.input_strides[dim] = in_strides[dim];
          if (static_cast<int64_t>(dim) == reduce_axis) {
            params.output_strides[dim] = 0;
          } else {
            params.output_strides[dim] =
                keepdims ? output_strides[dim] : output_strides[out_dim++];
          }
        }
        MusaReduceOp reduce_op = MusaReduceOp::Prod;
        if (mode == ReduceMode::kSum) reduce_op = MusaReduceOp::Sum;
        else if (mode == ReduceMode::kMean) reduce_op = MusaReduceOp::Mean;
        else if (mode == ReduceMode::kSumSquare) reduce_op = MusaReduceOp::SumSquare;
        return LaunchStatus(LaunchMusaReduceFloatKernel(
            input0.GetTensorData<float>(), y.GetTensorMutableData<float>(),
            params, reduce_op, nullptr));
      }
    }

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
      else if (mode == ReduceMode::kSumSquare)
        out[static_cast<size_t>(oo)] +=
            x[static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
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
