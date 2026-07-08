// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "shared_inc/kernel_memory.h"
#include "utils.h"

template <typename T>
std::span<const T> Span(const std::vector<uint8_t>& bytes) {
  return std::span<const T>(reinterpret_cast<const T*>(bytes.data()),
                            bytes.size() / sizeof(T));
}

template <typename T>
std::vector<T> ReadTyped(Ort::ConstValue value, musaStream_t stream = nullptr) {
  std::vector<uint8_t> bytes;
  Ort::ThrowOnError(CopyToHost(value, bytes, stream));
  std::vector<T> out(bytes.size() / sizeof(T));
  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }
  return out;
}

template <typename T>
OrtStatus* WriteTyped(Ort::UnownedValue value, const std::vector<T>& data,
                      musaStream_t stream = nullptr) {
  return CopyFromHost(value, data.data(), data.size() * sizeof(T), stream);
}

inline std::vector<int64_t> ReadIntTensor(Ort::KernelContext& ctx,
                                          size_t index) {
  Ort::ConstValue value = ctx.GetInput(index);
  musaStream_t stream = GetComputeStream(ctx);
  auto info = value.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value, stream);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> vals = ReadTyped<int32_t>(value, stream);
    return std::vector<int64_t>(vals.begin(), vals.end());
  }
  throw std::runtime_error("expected int32/int64 tensor");
}
