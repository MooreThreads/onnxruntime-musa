// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mublas.h>
#include <mudnncxx/mudnn.h>

#include <cstdlib>
#include <memory>

#include "shared_inc/op_kernel_common.h"
#include "math/gemm_post_kernels.h"

// Shared GEMM implementation used by Gemm and FusedGemm. MatMul keeps its
// own implementation because it also handles batched MatMul shapes.
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

inline bool ResolveTF32EnabledForGemm() {
  const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
  return tf32_env != nullptr && std::atoi(tf32_env) != 0;
}

inline OrtStatus* EnsureMudnnHandle(::musa::dnn::Handle** handle) {
  static thread_local std::unique_ptr<::musa::dnn::Handle> g_handle;
  if (!g_handle) {
    g_handle = std::make_unique<::musa::dnn::Handle>();
    auto status = g_handle->SetAllowTF32(ResolveTF32EnabledForGemm());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        "mudnn Handle SetAllowTF32 failed");
    }
  }
  *handle = g_handle.get();
  return nullptr;
}

inline bool MudnnTensorType(ONNXTensorElementDataType elem_type,
                            ::musa::dnn::Tensor::Type& mudnn_type) {
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      mudnn_type = ::musa::dnn::Tensor::Type::FLOAT;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      mudnn_type = ::musa::dnn::Tensor::Type::DOUBLE;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      mudnn_type = ::musa::dnn::Tensor::Type::HALF;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      mudnn_type = ::musa::dnn::Tensor::Type::BFLOAT16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      mudnn_type = ::musa::dnn::Tensor::Type::INT8;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      mudnn_type = ::musa::dnn::Tensor::Type::INT16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      mudnn_type = ::musa::dnn::Tensor::Type::INT32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      mudnn_type = ::musa::dnn::Tensor::Type::INT64;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      mudnn_type = ::musa::dnn::Tensor::Type::UINT8;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      mudnn_type = ::musa::dnn::Tensor::Type::UINT16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      mudnn_type = ::musa::dnn::Tensor::Type::UINT32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      mudnn_type = ::musa::dnn::Tensor::Type::UINT64;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      mudnn_type = ::musa::dnn::Tensor::Type::BOOL;
      return true;
    default:
      return false;
  }
}

inline bool SetMudnnTensor(::musa::dnn::Tensor& tensor, const void* data,
                           const std::vector<int64_t>& shape,
                           ONNXTensorElementDataType elem_type) {
  ::musa::dnn::Tensor::Type mudnn_type;
  if (!MudnnTensorType(elem_type, mudnn_type)) return false;
  if (tensor.SetAddr(data) != ::musa::dnn::Status::SUCCESS) return false;
  if (tensor.SetType(mudnn_type) != ::musa::dnn::Status::SUCCESS) return false;
  if (tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW) !=
      ::musa::dnn::Status::SUCCESS)
    return false;
  std::vector<int64_t> dims = shape.empty() ? std::vector<int64_t>{1} : shape;
  return tensor.SetNdInfo(static_cast<int64_t>(dims.size()), dims.data()) ==
         ::musa::dnn::Status::SUCCESS;
}

inline bool SetMudnnFloatTensor(::musa::dnn::Tensor& tensor, const void* data,
                                const std::vector<int64_t>& shape) {
  return SetMudnnTensor(tensor, data, shape,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

inline bool TryMudnnGemm(float* y_data, const float* a_data,
                         const float* b_data, const float* c_data,
                         const std::vector<int64_t>& a_shape,
                         const std::vector<int64_t>& b_shape,
                         const std::vector<int64_t>& c_shape,
                         const std::vector<int64_t>& out_shape, bool trans_a,
                         bool trans_b, float alpha, float beta,
                         bool has_bias) {
  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor a_tensor;
  ::musa::dnn::Tensor b_tensor;
  ::musa::dnn::Tensor y_tensor;
  if (!SetMudnnFloatTensor(a_tensor, a_data, a_shape) ||
      !SetMudnnFloatTensor(b_tensor, b_data, b_shape) ||
      !SetMudnnFloatTensor(y_tensor, y_data, out_shape)) {
    return false;
  }

  ::musa::dnn::MatMul matmul;
  if (matmul.SetTranspose(trans_a, trans_b) != ::musa::dnn::Status::SUCCESS)
    return false;
  if (matmul.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR) !=
      ::musa::dnn::Status::SUCCESS)
    return false;
  if (matmul.SetAlpha(static_cast<double>(alpha)) !=
      ::musa::dnn::Status::SUCCESS)
    return false;
  if (matmul.SetBeta(0.0) != ::musa::dnn::Status::SUCCESS) return false;
  if (matmul.SetGamma(0.0) != ::musa::dnn::Status::SUCCESS) return false;

  if (!has_bias || beta == 0.0f) {
    return matmul.Run(*handle, y_tensor, a_tensor, b_tensor) ==
           ::musa::dnn::Status::SUCCESS;
  }

  const int64_t m = out_shape[0];
  const int64_t n = out_shape[1];
  ::musa::dnn::Tensor c_tensor;
  ::musa::dnn::Tensor empty_tensor;

  if (c_shape.size() == 1 && c_shape[0] == n) {
    if (!SetMudnnFloatTensor(c_tensor, c_data, c_shape)) return false;
    if (matmul.SetGamma(static_cast<double>(beta)) !=
        ::musa::dnn::Status::SUCCESS)
      return false;
    return matmul.RunWithBiasAdd(*handle, y_tensor, a_tensor, b_tensor,
                                 y_tensor, c_tensor) ==
           ::musa::dnn::Status::SUCCESS;
  }

  if (c_shape.size() == 2 && c_shape[0] == 1 && c_shape[1] == n) {
    std::vector<int64_t> bias_shape = {n};
    if (!SetMudnnFloatTensor(c_tensor, c_data, bias_shape)) return false;
    if (matmul.SetGamma(static_cast<double>(beta)) !=
        ::musa::dnn::Status::SUCCESS)
      return false;
    return matmul.RunWithBiasAdd(*handle, y_tensor, a_tensor, b_tensor,
                                 y_tensor, c_tensor) ==
           ::musa::dnn::Status::SUCCESS;
  }

  if (c_shape.size() == 2 && c_shape[0] == m && c_shape[1] == n) {
    if (!SetMudnnFloatTensor(c_tensor, c_data, c_shape)) return false;
    if (matmul.SetBeta(static_cast<double>(beta)) !=
        ::musa::dnn::Status::SUCCESS)
      return false;
    return matmul.RunWithBiasAdd(*handle, y_tensor, a_tensor, b_tensor,
                                 c_tensor, empty_tensor) ==
           ::musa::dnn::Status::SUCCESS;
  }

  return false;
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

inline bool TryMapActivation(const std::string& activation, MusaUnaryOp& op) {
  if (activation == "Relu") {
    op = MusaUnaryOp::Relu;
    return true;
  }
  if (activation == "LeakyRelu") {
    op = MusaUnaryOp::LeakyRelu;
    return true;
  }
  if (activation == "Tanh") {
    op = MusaUnaryOp::Tanh;
    return true;
  }
  return false;
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
      std::vector<int64_t> c_shape = {1};
      const float* c_data = nullptr;
      const bool has_bias = ctx.GetInputCount() > 2;
      bool bias_is_gpu = true;
      if (has_bias) {
        auto c_value = ctx.GetInput(2);
        auto c_info = c_value.GetTensorTypeAndShapeInfo();
        if (c_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED, "Gemm bias only supports float tensors");
        }
        c_shape = c_info.GetShape();
        std::vector<int64_t> broadcast_shape =
            BroadcastShape(c_shape, out_shape);
        if (broadcast_shape != out_shape) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT, "Gemm bias broadcast shape mismatch");
        }
        c_data = c_value.GetTensorData<float>();
        bias_is_gpu = IsGpuMemory(c_value.GetTensorMemoryInfo());
      }

      if (bias_is_gpu &&
          TryMudnnGemm(y_data, a_data, b_data, c_data, a_shape, b_shape,
                       c_shape, out_shape, trans_a, trans_b, alpha, beta,
                       has_bias)) {
        if (activation.empty()) {
          return nullptr;
        }

        MusaUnaryOp activation_op = MusaUnaryOp::Relu;
        if (TryMapActivation(activation, activation_op)) {
          MusaBroadcastParams params =
              MakeBroadcastParams(out_shape, out_shape, {1});
          return LaunchStatus(LaunchMusaGemmPostFloatKernel(
              y_data, nullptr, params, false, 0.0f, activation_op, true,
              activation_alpha, nullptr));
        }

        std::vector<float> out(static_cast<size_t>(m * n));
        musaError_t sync_status =
            musaMemcpy(out.data(), y_data, out.size() * sizeof(float),
                       musaMemcpyDeviceToHost);
        if (sync_status != musaSuccess) {
          return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                            MusaErrorString(sync_status));
        }
        ApplyActivation(out, activation, activation_alpha);
        return WriteTyped<float>(y, out);
      }

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
      if (ctx.GetInputCount() <= 2 && activation.empty()) {
        return nullptr;
      }

      MusaUnaryOp activation_op = MusaUnaryOp::Relu;
      const bool has_activation = !activation.empty();
      const bool activation_supported =
          !has_activation || TryMapActivation(activation, activation_op);
      if (activation_supported && (!has_bias || bias_is_gpu)) {
        if (CanUseBroadcastKernel(out_shape, out_shape, c_shape)) {
          MusaBroadcastParams params =
              MakeBroadcastParams(out_shape, out_shape, c_shape);
          return LaunchStatus(LaunchMusaGemmPostFloatKernel(
              y_data, c_data, params, has_bias, beta, activation_op,
              has_activation, activation_alpha, nullptr));
        }
      }

      std::vector<float> out(static_cast<size_t>(m * n));
      musaError_t sync_status =
          musaMemcpy(out.data(), y_data, out.size() * sizeof(float),
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
        int64_t a_off =
            trans_a ? kk * a_shape[1] + row : row * a_shape[1] + kk;
        int64_t b_off =
            trans_b ? col * b_shape[1] + kk : kk * b_shape[1] + col;
        sum += a[static_cast<size_t>(a_off)] *
               b[static_cast<size_t>(b_off)];
      }
      out[static_cast<size_t>(row * n + col)] = alpha * sum;
    }
  }

  if (ctx.GetInputCount() > 2) {
    auto c_value = ctx.GetInput(2);
    auto c_info = c_value.GetTensorTypeAndShapeInfo();
    if (c_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Gemm bias only supports float tensors");
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
