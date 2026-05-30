// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mublas.h>

#include "common/op_kernel_common.h"

// Shared mUBLAS handle + Gemm implementation used by the Gemm / FusedGemm
// kernels. MatMul keeps its own copy in matmul.cc.
inline OrtStatus* EnsureMublasHandle(mublasHandle_t* handle) {
  static thread_local mublasHandle_t g_handle = nullptr;
  if (g_handle == nullptr) {
    mublasStatus status = mublasCreate(&g_handle);
    if (status != MUBLAS_STATUS_SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasCreate failed");
    }
  }
  *handle = g_handle;
  return nullptr;
}

inline void ApplyActivation(std::vector<float>& values,
                            const std::string& activation,
                            float activation_alpha) {
  if (activation.empty()) {
    return;
  }
  if (activation == "Relu") {
    for (float& v : values) v = std::max(0.0f, v);
    return;
  }
  if (activation == "LeakyRelu") {
    for (float& v : values) v = v >= 0.0f ? v : activation_alpha * v;
    return;
  }
  if (activation == "Tanh") {
    for (float& v : values) v = std::tanh(v);
  }
}

inline OrtStatus* GemmCompute(Ort::KernelContext& ctx, bool trans_a,
                              bool trans_b, float alpha, float beta,
                              const std::string& activation,
                              float activation_alpha) {
  auto a_info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto b_info = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  if (a_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gemm only supports float tensors");
  }

  std::vector<int64_t> a_shape = a_info.GetShape();
  std::vector<int64_t> b_shape = b_info.GetShape();
  if (a_shape.size() != 2 || b_shape.size() != 2) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gemm requires rank-2 inputs");
  }

  int64_t m = trans_a ? a_shape[1] : a_shape[0];
  int64_t k = trans_a ? a_shape[0] : a_shape[1];
  int64_t kb = trans_b ? b_shape[1] : b_shape[0];
  int64_t n = trans_b ? b_shape[0] : b_shape[1];
  if (k != kb) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Gemm K dimension mismatch");
  }

  std::vector<int64_t> out_shape = {m, n};
  bool can_use_mublas = IsGpuMemory(ctx.GetInput(0).GetTensorMemoryInfo()) &&
                        IsGpuMemory(ctx.GetInput(1).GetTensorMemoryInfo());
  if (can_use_mublas) {
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    can_use_mublas = IsGpuMemory(y.GetTensorMemoryInfo());
    if (can_use_mublas) {
      if (m > INT32_MAX || k > INT32_MAX || n > INT32_MAX) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT, "Gemm dimensions exceed int32 mublas limits");
      }
      const float* a_data = ctx.GetInput(0).GetTensorData<float>();
      const float* b_data = ctx.GetInput(1).GetTensorData<float>();
      float* y_data = y.GetTensorMutableData<float>();
      mublasHandle_t handle = nullptr;
      RETURN_IF_ERROR(EnsureMublasHandle(&handle));
      mublasOperation_t op_a = trans_a ? MUBLAS_OP_T : MUBLAS_OP_N;
      mublasOperation_t op_b = trans_b ? MUBLAS_OP_T : MUBLAS_OP_N;
      int lda = static_cast<int>(a_shape[1]);
      int ldb = static_cast<int>(b_shape[1]);
      int mi = static_cast<int>(m);
      int ki = static_cast<int>(k);
      int ni = static_cast<int>(n);
      float zero = 0.0f;
      mublasStatus status =
          mublasSgemm(handle, op_b, op_a, ni, mi, ki, &alpha, b_data, ldb,
                      a_data, lda, &zero, y_data, ni);
      if (status != MUBLAS_STATUS_SUCCESS) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasSgemm failed");
      }
      musaError_t sync_status = musaDeviceSynchronize();
      if (sync_status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                          MusaErrorString(sync_status));
      }
      if (ctx.GetInputCount() <= 2 && activation.empty()) {
        return nullptr;
      }

      std::vector<float> out(static_cast<size_t>(m * n));
      sync_status = musaMemcpy(out.data(), y_data, out.size() * sizeof(float),
                               musaMemcpyDeviceToHost);
      if (sync_status != musaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                          MusaErrorString(sync_status));
      }
      if (ctx.GetInputCount() > 2) {
        auto c_value = ctx.GetInput(2);
        auto c_info = c_value.GetTensorTypeAndShapeInfo();
        if (c_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "Gemm bias only supports float tensors");
        }
        std::vector<int64_t> c_shape = c_info.GetShape();
        std::vector<float> c = ReadTyped<float>(c_value);
        std::vector<int64_t> broadcast_shape =
            BroadcastShape(c_shape, out_shape);
        if (broadcast_shape != out_shape) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "Gemm bias broadcast shape mismatch");
        }
        auto c_strides = Strides(c_shape);
        for (int64_t i = 0; i < NumElements(out_shape); ++i) {
          auto coord = Coordinates(i, out_shape);
          int64_t c_off = BroadcastOffset(coord, c_shape, c_strides);
          out[static_cast<size_t>(i)] += beta * c[static_cast<size_t>(c_off)];
        }
      }
      ApplyActivation(out, activation, activation_alpha);
      return WriteTyped<float>(y, out);
    }
  }

  std::vector<float> a = ReadTyped<float>(ctx.GetInput(0));
  std::vector<float> b = ReadTyped<float>(ctx.GetInput(1));
  std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        int64_t a_off = trans_a ? kk * a_shape[1] + row : row * a_shape[1] + kk;
        int64_t b_off = trans_b ? col * b_shape[1] + kk : kk * b_shape[1] + col;
        sum += a[static_cast<size_t>(a_off)] * b[static_cast<size_t>(b_off)];
      }
      out[static_cast<size_t>(row * n + col)] = alpha * sum;
    }
  }

  if (ctx.GetInputCount() > 2) {
    auto c_value = ctx.GetInput(2);
    auto c_info = c_value.GetTensorTypeAndShapeInfo();
    if (c_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(
          ORT_NOT_IMPLEMENTED, "Gemm bias only supports float tensors");
    }
    std::vector<int64_t> c_shape = c_info.GetShape();
    std::vector<float> c = ReadTyped<float>(c_value);
    std::vector<int64_t> broadcast_shape = BroadcastShape(c_shape, out_shape);
    if (broadcast_shape != out_shape) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Gemm bias broadcast shape mismatch");
    }
    auto c_strides = Strides(c_shape);
    for (int64_t i = 0; i < NumElements(out_shape); ++i) {
      auto coord = Coordinates(i, out_shape);
      int64_t c_off = BroadcastOffset(coord, c_shape, c_strides);
      out[static_cast<size_t>(i)] += beta * c[static_cast<size_t>(c_off)];
    }
  }

  ApplyActivation(out, activation, activation_alpha);
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<float>(y, out);
}
