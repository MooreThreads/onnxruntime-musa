// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "math/binary_elementwise_ops_impl.h"
#include "math/unary_elementwise_ops_impl.h"
#include "shared_inc/kernel_element_types.h"
#include "shared_inc/kernel_memory.h"
#include "shared_inc/kernel_shape_utils.h"
#include "shared_inc/kernel_typed_io.h"
#include "utils.h"

template <typename T, typename Fn>
OrtStatus* BinaryCompute(Ort::KernelContext& ctx,
                         const std::vector<int64_t>& shape0,
                         const std::vector<int64_t>& shape1, Fn fn) {
  musaStream_t stream = GetComputeStream(ctx);
  std::vector<T> a = ReadTyped<T>(ctx.GetInput(0), stream);
  std::vector<T> b = ReadTyped<T>(ctx.GetInput(1), stream);
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  int64_t total = NumElements(out_shape);
  std::vector<T> out(static_cast<size_t>(total));

  auto s0 = Strides(shape0);
  auto s1 = Strides(shape1);
  for (int64_t i = 0; i < total; ++i) {
    auto coord = Coordinates(i, out_shape);
    int64_t o0 = BroadcastOffset(coord, shape0, s0);
    int64_t o1 = BroadcastOffset(coord, shape1, s1);
    out[static_cast<size_t>(i)] =
        fn(a[static_cast<size_t>(o0)], b[static_cast<size_t>(o1)]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<T>(y, out, stream);
}

template <typename T, typename Fn>
OrtStatus* UnaryCompute(Ort::KernelContext& ctx,
                        const std::vector<int64_t>& shape, Fn fn) {
  musaStream_t stream = GetComputeStream(ctx);
  std::vector<T> x = ReadTyped<T>(ctx.GetInput(0), stream);
  std::vector<T> y_data(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    y_data[i] = fn(x[i]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  return WriteTyped<T>(y, y_data, stream);
}

template <typename T, typename Fn>
OrtStatus* BinaryCompute(Ort::KernelContext& ctx,
                         const std::vector<int64_t>& shape0,
                         const std::vector<int64_t>& shape1, Fn fn,
                         MusaBinaryOp device_op) {
  if constexpr (std::is_same_v<T, float>) {
    std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
    Ort::ConstValue lhs = ctx.GetInput(0);
    Ort::ConstValue rhs = ctx.GetInput(1);
    if (IsGpuMemory(lhs.GetTensorMemoryInfo()) &&
        IsGpuMemory(rhs.GetTensorMemoryInfo()) &&
        CanUseBroadcastKernel(out_shape, shape0, shape1)) {
      Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
      if (IsGpuMemory(y.GetTensorMemoryInfo())) {
        MusaBroadcastParams params =
            MakeBroadcastParams(out_shape, shape0, shape1);
        return LaunchStatus(LaunchMusaBinaryFloatKernel(
            lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
            y.GetTensorMutableData<float>(), params, device_op,
            GetComputeStream(ctx)));
      }
    }
  }
  return BinaryCompute<T>(ctx, shape0, shape1, fn);
}

template <typename T, typename Fn>
OrtStatus* UnaryCompute(Ort::KernelContext& ctx,
                        const std::vector<int64_t>& shape, Fn fn,
                        MusaUnaryOp device_op, float alpha = 0.0f) {
  if constexpr (std::is_same_v<T, float>) {
    Ort::ConstValue input = ctx.GetInput(0);
    if (IsGpuMemory(input.GetTensorMemoryInfo())) {
      Ort::UnownedValue y = ctx.GetOutput(0, shape);
      if (IsGpuMemory(y.GetTensorMemoryInfo())) {
        return LaunchStatus(LaunchMusaUnaryFloatKernel(
            input.GetTensorData<float>(), y.GetTensorMutableData<float>(),
            NumElements(shape), device_op, alpha, GetComputeStream(ctx)));
      }
    }
  }
  return UnaryCompute<T>(ctx, shape, fn);
}

inline OrtStatus* BinaryDeviceCompute(Ort::KernelContext& ctx,
                                      const std::vector<int64_t>& shape0,
                                      const std::vector<int64_t>& shape1,
                                      ONNXTensorElementDataType elem_type,
                                      MusaBinaryOp device_op,
                                      const char* op_name) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  if (std::getenv("MUSA_EP_TRACE_KERNELS") != nullptr) {
    std::string message = "MUSA_BINARY ";
    message += op_name;
    message += " lhs=";
    AppendShapeForError(message, shape0);
    message +=
        IsGpuMemory(lhs.GetTensorMemoryInfo()) ? " lhs_gpu=1" : " lhs_gpu=0";
    message += " rhs=";
    AppendShapeForError(message, shape1);
    message +=
        IsGpuMemory(rhs.GetTensorMemoryInfo()) ? " rhs_gpu=1" : " rhs_gpu=0";
    message += " out=";
    AppendShapeForError(message, out_shape);
    message += " total=" + std::to_string(NumElements(out_shape));
    char ptr_buffer[160];
    std::snprintf(ptr_buffer, sizeof(ptr_buffer), " lhs_ptr=%p rhs_ptr=%p",
                  lhs.GetTensorRawData(), rhs.GetTensorRawData());
    message += ptr_buffer;
    std::fprintf(stderr, "%s\n", message.c_str());
    std::fflush(stderr);
  }
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !CanUseBroadcastKernel(out_shape, shape0, shape1)) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  musaError_t status =
      LaunchMusaBinaryKernel(lhs.GetTensorRawData(), rhs.GetTensorRawData(),
                             y.GetTensorMutableRawData(),
                             MakeBroadcastParams(out_shape, shape0, shape1),
                             device_op, musa_elem_type, GetComputeStream(ctx));
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }
  return LaunchStatus(status);
}

inline OrtStatus* UnaryDeviceCompute(Ort::KernelContext& ctx,
                                     const std::vector<int64_t>& shape,
                                     ONNXTensorElementDataType elem_type,
                                     MusaUnaryOp device_op, const char* op_name,
                                     float alpha = 0.0f) {
  Ort::ConstValue input = ctx.GetInput(0);
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(input.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  musaError_t status = LaunchMusaUnaryKernel(
      input.GetTensorRawData(), y.GetTensorMutableRawData(), NumElements(shape),
      device_op, alpha, musa_elem_type, GetComputeStream(ctx));
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }
  return LaunchStatus(status);
}

inline bool OutputEmptyTensorIfNeeded(Ort::KernelContext& ctx,
                                      const std::vector<int64_t>& shape) {
  if (NumElements(shape) != 0) {
    return false;
  }
  ctx.GetOutput(0, shape);
  return true;
}

template <typename T>
inline bool CompareHostValue(T lhs, T rhs, MusaCompareOp op) {
  switch (op) {
    case MusaCompareOp::Equal:
      return lhs == rhs;
    case MusaCompareOp::Greater:
      return lhs > rhs;
    case MusaCompareOp::GreaterOrEqual:
      return lhs >= rhs;
    case MusaCompareOp::Less:
      return lhs < rhs;
    case MusaCompareOp::LessOrEqual:
      return lhs <= rhs;
  }
  return false;
}

template <typename T>
OrtStatus* CompareCpuMetadataTyped(Ort::KernelContext& ctx,
                                   const std::vector<int64_t>& shape0,
                                   const std::vector<int64_t>& shape1,
                                   const std::vector<int64_t>& out_shape,
                                   MusaCompareOp device_op,
                                   musaStream_t stream) {
  const int64_t total = NumElements(out_shape);
  if (total > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Compare CPU metadata path only supports small shape tensors");
  }
  std::vector<T> lhs = ReadTyped<T>(ctx.GetInput(0), stream);
  std::vector<T> rhs = ReadTyped<T>(ctx.GetInput(1), stream);
  std::vector<uint8_t> out(static_cast<size_t>(total));
  auto lhs_strides = Strides(shape0);
  auto rhs_strides = Strides(shape1);
  for (int64_t i = 0; i < total; ++i) {
    auto coord = Coordinates(i, out_shape);
    int64_t lhs_offset = BroadcastOffset(coord, shape0, lhs_strides);
    int64_t rhs_offset = BroadcastOffset(coord, shape1, rhs_strides);
    out[static_cast<size_t>(i)] =
        CompareHostValue(lhs[static_cast<size_t>(lhs_offset)],
                         rhs[static_cast<size_t>(rhs_offset)], device_op)
            ? 1
            : 0;
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<uint8_t>(y, out, stream);
}

inline OrtStatus* CompareCpuMetadata(Ort::KernelContext& ctx,
                                     const std::vector<int64_t>& shape0,
                                     const std::vector<int64_t>& shape1,
                                     const std::vector<int64_t>& out_shape,
                                     ONNXTensorElementDataType elem_type,
                                     MusaCompareOp device_op,
                                     musaStream_t stream) {
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return CompareCpuMetadataTyped<int64_t>(ctx, shape0, shape1, out_shape,
                                            device_op, stream);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return CompareCpuMetadataTyped<int32_t>(ctx, shape0, shape1, out_shape,
                                            device_op, stream);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL &&
      device_op == MusaCompareOp::Equal) {
    return CompareCpuMetadataTyped<uint8_t>(ctx, shape0, shape1, out_shape,
                                            device_op, stream);
  }
  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Compare CPU metadata path only supports int32/int64 tensors");
}

inline OrtStatus* CompareDeviceCompute(Ort::KernelContext& ctx,
                                       const std::vector<int64_t>& shape0,
                                       const std::vector<int64_t>& shape1,
                                       ONNXTensorElementDataType elem_type,
                                       MusaCompareOp device_op,
                                       const char* op_name) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !CanUseBroadcastKernel(out_shape, shape0, shape1)) {
    if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
        !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
      return CompareCpuMetadata(ctx, shape0, shape1, out_shape, elem_type,
                                device_op, GetComputeStream(ctx));
    }
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }

  musaError_t status =
      LaunchMusaCompareKernel(lhs.GetTensorRawData(), rhs.GetTensorRawData(),
                              y.GetTensorMutableData<uint8_t>(),
                              MakeBroadcastParams(out_shape, shape0, shape1),
                              device_op, musa_elem_type, GetComputeStream(ctx));
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus(op_name, elem_type);
  }
  return LaunchStatus(status);
}
