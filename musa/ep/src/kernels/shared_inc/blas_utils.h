// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mublas.h>
#include <mudnncxx/mudnn.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>

#include "math/gemm_post_kernels.h"
#include "math/matmul.h"
#include "shared_inc/op_kernel_common.h"

// Shared GEMM implementation used by Gemm and FusedGemm. MatMul keeps its
// own implementation because it also handles batched MatMul shapes.
inline OrtStatus* EnsureMublasHandle(mublasHandle_t* handle,
                                     musaStream_t stream = nullptr) {
  static thread_local mublasHandle_t g_handle = nullptr;
  static thread_local musaStream_t g_stream = nullptr;
  if (g_handle != nullptr && g_stream != stream) {
    mublasDestroy(g_handle);
    g_handle = nullptr;
  }
  if (g_handle == nullptr) {
    mublasStatus status = mublasCreate(&g_handle);
    if (status != MUBLAS_STATUS_SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasCreate failed");
    }
    g_stream = stream;
  }
  mublasStatus stream_status =
      mublasSetStream(g_handle, reinterpret_cast<MUstream>(stream));
  if (stream_status != MUBLAS_STATUS_SUCCESS) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasSetStream failed");
  }
  *handle = g_handle;
  return nullptr;
}

inline bool ResolveTF32EnabledForGemm() {
  const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
  return tf32_env != nullptr && std::atoi(tf32_env) != 0;
}

inline OrtStatus* EnsureMudnnHandle(::musa::dnn::Handle** handle,
                                    musaStream_t stream = nullptr) {
  static thread_local std::unique_ptr<::musa::dnn::Handle> g_handle;
  static thread_local musaStream_t g_stream = nullptr;
  if (g_handle && g_stream != stream) {
    g_handle.reset();
  }
  if (!g_handle) {
    g_handle = std::make_unique<::musa::dnn::Handle>();
    g_stream = stream;
    auto status = g_handle->SetAllowTF32(ResolveTF32EnabledForGemm());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        "mudnn Handle SetAllowTF32 failed");
    }
  }
  auto stream_status = g_handle->SetStream(stream);
  if (stream_status != ::musa::dnn::Status::SUCCESS) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "mudnn Handle SetStream failed");
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

inline void AppendShapeKey(std::string& key,
                           const std::vector<int64_t>& shape) {
  key.push_back('[');
  for (int64_t dim : shape) {
    key += std::to_string(dim);
    key.push_back(',');
  }
  key.push_back(']');
}

inline std::string MudnnGemmKey(const std::vector<int64_t>& a_shape,
                                const std::vector<int64_t>& b_shape,
                                const std::vector<int64_t>& c_shape,
                                const std::vector<int64_t>& out_shape,
                                bool trans_a, bool trans_b, float alpha,
                                float beta, bool has_bias,
                                ONNXTensorElementDataType elem_type) {
  std::string key = std::to_string(static_cast<int>(elem_type));
  key += trans_a ? "|ta1" : "|ta0";
  key += trans_b ? "|tb1" : "|tb0";
  key += "|a" + std::to_string(alpha);
  key += "|b" + std::to_string(beta);
  key += has_bias ? "|bias1" : "|bias0";
  AppendShapeKey(key, a_shape);
  AppendShapeKey(key, b_shape);
  AppendShapeKey(key, c_shape);
  AppendShapeKey(key, out_shape);
  return key;
}

inline std::unordered_set<std::string>& MudnnGemmUnsupportedKeys() {
  static thread_local std::unordered_set<std::string> keys;
  return keys;
}

inline bool MublasDataType(ONNXTensorElementDataType elem_type,
                           musaDataType_t& data_type,
                           mublasComputeType_t& compute_type) {
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      data_type = MUSA_R_32F;
      compute_type = MUBLAS_COMPUTE_32F;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      data_type = MUSA_R_64F;
      compute_type = MUBLAS_COMPUTE_64F;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      data_type = MUSA_R_16F;
      compute_type = MUBLAS_COMPUTE_32F;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      data_type = MUSA_R_16BF;
      compute_type = MUBLAS_COMPUTE_32F;
      return true;
    default:
      return false;
  }
}

inline mublasStatus MublasGemmEx(mublasHandle_t handle,
                                 mublasOperation_t trans_a,
                                 mublasOperation_t trans_b, int m, int n, int k,
                                 double alpha, const void* a, int lda,
                                 const void* b, int ldb, double beta, void* c,
                                 int ldc, ONNXTensorElementDataType elem_type) {
  musaDataType_t data_type;
  mublasComputeType_t compute_type;
  if (!MublasDataType(elem_type, data_type, compute_type)) {
    return MUBLAS_STATUS_NOT_SUPPORTED;
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    const double alpha64 = alpha;
    const double beta64 = beta;
    return mublasGemmEx(handle, trans_a, trans_b, m, n, k, &alpha64, a,
                        data_type, lda, b, data_type, ldb, &beta64, c,
                        data_type, ldc, compute_type, MUBLAS_GEMM_DEFAULT);
  }
  const float alpha32 = static_cast<float>(alpha);
  const float beta32 = static_cast<float>(beta);
  return mublasGemmEx(handle, trans_a, trans_b, m, n, k, &alpha32, a, data_type,
                      lda, b, data_type, ldb, &beta32, c, data_type, ldc,
                      compute_type, MUBLAS_GEMM_DEFAULT);
}

inline mublasStatus MublasGemmStridedBatchedEx(
    mublasHandle_t handle, mublasOperation_t trans_a, mublasOperation_t trans_b,
    int m, int n, int k, double alpha, const void* a, int lda,
    long long int stride_a, const void* b, int ldb, long long int stride_b,
    double beta, void* c, int ldc, long long int stride_c, int batch_count,
    ONNXTensorElementDataType elem_type) {
  musaDataType_t data_type;
  mublasComputeType_t compute_type;
  if (!MublasDataType(elem_type, data_type, compute_type)) {
    return MUBLAS_STATUS_NOT_SUPPORTED;
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    const double alpha64 = alpha;
    const double beta64 = beta;
    return mublasGemmStridedBatchedEx(
        handle, trans_a, trans_b, m, n, k, &alpha64, a, data_type, lda,
        stride_a, b, data_type, ldb, stride_b, &beta64, c, data_type, ldc,
        stride_c, batch_count, compute_type, MUBLAS_GEMM_DEFAULT);
  }
  const float alpha32 = static_cast<float>(alpha);
  const float beta32 = static_cast<float>(beta);
  return mublasGemmStridedBatchedEx(
      handle, trans_a, trans_b, m, n, k, &alpha32, a, data_type, lda, stride_a,
      b, data_type, ldb, stride_b, &beta32, c, data_type, ldc, stride_c,
      batch_count, compute_type, MUBLAS_GEMM_DEFAULT);
}

inline bool TryMudnnGemm(
    void* y_data, const void* a_data, const void* b_data, const void* c_data,
    const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
    const std::vector<int64_t>& c_shape, const std::vector<int64_t>& out_shape,
    bool trans_a, bool trans_b, float alpha, float beta, bool has_bias,
    ONNXTensorElementDataType elem_type, musaStream_t stream = nullptr) {
  const std::string key =
      MudnnGemmKey(a_shape, b_shape, c_shape, out_shape, trans_a, trans_b,
                   alpha, beta, has_bias, elem_type);
  auto& unsupported_keys = MudnnGemmUnsupportedKeys();
  if (unsupported_keys.count(key) != 0) {
    return false;
  }

  // muDNN MatMul::RunWithBiasAdd reports NOT_SUPPORTED for DOUBLE on this
  // stack. Avoid issuing the unsupported library call; the caller will use
  // muBLAS/MatMul followed by the device post kernel for bias/activation.
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE && has_bias &&
      beta != 0.0f) {
    unsupported_keys.insert(key);
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, stream);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor a_tensor;
  ::musa::dnn::Tensor b_tensor;
  ::musa::dnn::Tensor y_tensor;
  if (!SetMudnnTensor(a_tensor, a_data, a_shape, elem_type) ||
      !SetMudnnTensor(b_tensor, b_data, b_shape, elem_type) ||
      !SetMudnnTensor(y_tensor, y_data, out_shape, elem_type)) {
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
    const bool ok = matmul.Run(*handle, y_tensor, a_tensor, b_tensor) ==
                    ::musa::dnn::Status::SUCCESS;
    if (!ok) unsupported_keys.insert(key);
    return ok;
  }

  const int64_t m = out_shape[0];
  const int64_t n = out_shape[1];
  ::musa::dnn::Tensor c_tensor;
  ::musa::dnn::Tensor empty_tensor;

  if (c_shape.size() == 1 && c_shape[0] == n) {
    if (!SetMudnnTensor(c_tensor, c_data, c_shape, elem_type)) return false;
    if (matmul.SetGamma(static_cast<double>(beta)) !=
        ::musa::dnn::Status::SUCCESS)
      return false;
    const bool ok =
        matmul.RunWithBiasAdd(*handle, y_tensor, a_tensor, b_tensor, y_tensor,
                              c_tensor) == ::musa::dnn::Status::SUCCESS;
    if (!ok) unsupported_keys.insert(key);
    return ok;
  }

  if (c_shape.size() == 2 && c_shape[0] == 1 && c_shape[1] == n) {
    std::vector<int64_t> bias_shape = {n};
    if (!SetMudnnTensor(c_tensor, c_data, bias_shape, elem_type)) return false;
    if (matmul.SetGamma(static_cast<double>(beta)) !=
        ::musa::dnn::Status::SUCCESS)
      return false;
    const bool ok =
        matmul.RunWithBiasAdd(*handle, y_tensor, a_tensor, b_tensor, y_tensor,
                              c_tensor) == ::musa::dnn::Status::SUCCESS;
    if (!ok) unsupported_keys.insert(key);
    return ok;
  }

  if (c_shape.size() == 2 && c_shape[0] == m && c_shape[1] == n) {
    if (!SetMudnnTensor(c_tensor, c_data, c_shape, elem_type)) return false;
    if (matmul.SetBeta(static_cast<double>(beta)) !=
        ::musa::dnn::Status::SUCCESS)
      return false;
    const bool ok =
        matmul.RunWithBiasAdd(*handle, y_tensor, a_tensor, b_tensor, c_tensor,
                              empty_tensor) == ::musa::dnn::Status::SUCCESS;
    if (!ok) unsupported_keys.insert(key);
    return ok;
  }

  unsupported_keys.insert(key);
  return false;
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
  if (activation == "Sigmoid") {
    op = MusaUnaryOp::Sigmoid;
    return true;
  }
  return false;
}

struct GemmShapeInfo {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t lda = 0;
  int64_t ldb = 0;
  std::vector<int64_t> out_shape;
};

inline OrtStatus* ResolveGemmShape(const std::vector<int64_t>& a_shape,
                                   const std::vector<int64_t>& b_shape,
                                   bool trans_a, bool trans_b,
                                   GemmShapeInfo& shape_info) {
  if (a_shape.size() != 2 || b_shape.size() != 2) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gemm requires rank-2 inputs");
  }

  shape_info.m = trans_a ? a_shape[1] : a_shape[0];
  shape_info.k = trans_a ? a_shape[0] : a_shape[1];
  shape_info.lda = a_shape[1];
  const int64_t kb = trans_b ? b_shape[1] : b_shape[0];
  shape_info.n = trans_b ? b_shape[0] : b_shape[1];
  shape_info.ldb = b_shape[1];
  shape_info.out_shape = {shape_info.m, shape_info.n};
  if (shape_info.k != kb) {
    std::string message = "Gemm K dimension mismatch: A=";
    AppendShapeKey(message, a_shape);
    message += " B=";
    AppendShapeKey(message, b_shape);
    message += trans_a ? " transA=1" : " transA=0";
    message += trans_b ? " transB=1" : " transB=0";
    message += " K=" + std::to_string(shape_info.k);
    message += " KB=" + std::to_string(kb);
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }
  return nullptr;
}

inline OrtStatus* GemmCompute(Ort::KernelContext& ctx, bool trans_a,
                              bool trans_b, float alpha, float beta,
                              const std::string& activation,
                              float activation_alpha) {
  musaStream_t stream = GetComputeStream(ctx);
  auto a_info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto b_info = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  const auto elem_type = a_info.GetElementType();
  musaDataType_t unused_data_type;
  mublasComputeType_t unused_compute_type;
  if (!MublasDataType(elem_type, unused_data_type, unused_compute_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported Gemm dtype");
  }
  if (b_info.GetElementType() != elem_type) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Gemm input dtypes must match");
  }

  std::vector<int64_t> a_shape = a_info.GetShape();
  std::vector<int64_t> b_shape = b_info.GetShape();
  GemmShapeInfo shape_info;
  RETURN_IF_ERROR(
      ResolveGemmShape(a_shape, b_shape, trans_a, trans_b, shape_info));
  const int64_t m = shape_info.m;
  const int64_t k = shape_info.k;
  const int64_t n = shape_info.n;
  const std::vector<int64_t>& out_shape = shape_info.out_shape;
  if (!IsGpuMemory(ctx.GetInput(0).GetTensorMemoryInfo()) ||
      !IsGpuMemory(ctx.GetInput(1).GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gemm requires MUSA inputs");
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gemm requires MUSA output");
  }

  std::vector<int64_t> c_shape = {1};
  const void* c_data = nullptr;
  const bool has_bias = ctx.GetInputCount() > 2;
  bool bias_is_gpu = true;
  if (has_bias) {
    auto c_value = ctx.GetInput(2);
    auto c_info = c_value.GetTensorTypeAndShapeInfo();
    if (c_info.GetElementType() != elem_type) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Gemm bias dtype must match inputs");
    }
    c_shape = c_info.GetShape();
    std::vector<int64_t> broadcast_shape = BroadcastShape(c_shape, out_shape);
    if (broadcast_shape != out_shape) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Gemm bias broadcast shape mismatch");
    }
    c_data = c_value.GetTensorRawData();
    bias_is_gpu = IsGpuMemory(c_value.GetTensorMemoryInfo());
  }

  if (has_bias && !bias_is_gpu) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gemm bias requires MUSA input");
  }

  if (NumElements(out_shape) == 0) {
    return nullptr;
  }

  MusaUnaryOp activation_op = MusaUnaryOp::Relu;
  const bool has_activation = !activation.empty();
  if (has_activation && !TryMapActivation(activation, activation_op)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "unsupported FusedGemm activation for MUSA post kernel");
  }

  const void* a_data = ctx.GetInput(0).GetTensorRawData();
  const void* b_data = ctx.GetInput(1).GetTensorRawData();
  void* y_data = y.GetTensorMutableRawData();

  if (bias_is_gpu && TryMudnnGemm(y_data, a_data, b_data, c_data, a_shape,
                                  b_shape, c_shape, out_shape, trans_a, trans_b,
                                  alpha, beta, has_bias, elem_type, stream)) {
    if (!has_activation) {
      return nullptr;
    }
    MusaElementType musa_elem_type;
    if (!ToMusaElementType(elem_type, musa_elem_type)) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "unsupported Gemm dtype");
    }
    MusaBroadcastParams params = MakeBroadcastParams(out_shape, out_shape, {1});
    return LaunchStatus(LaunchMusaGemmPostKernel(
        y_data, nullptr, params, false, 0.0f, activation_op, true,
        activation_alpha, musa_elem_type, stream));
  }

  bool gemm_done = false;
  if (m <= INT32_MAX && k <= INT32_MAX && n <= INT32_MAX) {
    mublasHandle_t handle = nullptr;
    RETURN_IF_ERROR(EnsureMublasHandle(&handle, stream));
    mublasOperation_t op_a = trans_a ? MUBLAS_OP_T : MUBLAS_OP_N;
    mublasOperation_t op_b = trans_b ? MUBLAS_OP_T : MUBLAS_OP_N;
    int lda = static_cast<int>(shape_info.lda);
    int ldb = static_cast<int>(shape_info.ldb);
    int mi = static_cast<int>(m);
    int ki = static_cast<int>(k);
    int ni = static_cast<int>(n);
    mublasStatus status =
        MublasGemmEx(handle, op_b, op_a, ni, mi, ki, alpha, b_data, ldb, a_data,
                     lda, 0.0, y_data, ni, elem_type);
    gemm_done = status == MUBLAS_STATUS_SUCCESS;
  }
  if (!gemm_done) {
    RETURN_IF_ERROR(ComputeMusaMatMulDevice(
        a_data, b_data, y_data, elem_type, a_shape, b_shape, out_shape, trans_a,
        trans_b, false, false, alpha, stream));
  }

  if (!has_bias && !has_activation) {
    return nullptr;
  }

  if (!CanUseBroadcastKernel(out_shape, out_shape, c_shape)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Gemm bias broadcast rank exceeds MUSA kernel limit");
  }
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported Gemm dtype");
  }
  MusaBroadcastParams params =
      MakeBroadcastParams(out_shape, out_shape, c_shape);
  return LaunchStatus(LaunchMusaGemmPostKernel(
      y_data, c_data, params, has_bias, beta, activation_op, has_activation,
      activation_alpha, musa_elem_type, stream));
}
