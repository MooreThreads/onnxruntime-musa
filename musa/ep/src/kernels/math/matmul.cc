#include "matmul.h"

#include <mublas.h>
#include <mudnncxx/mudnn.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#include "matmul_batched_kernels.h"
#include "runtime/ep_musa_utils.h"

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint(
        "T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    MatMul)

namespace {
thread_local mublasHandle_t g_handle = nullptr;
thread_local std::unique_ptr<::musa::dnn::Handle> g_mudnn_handle;

OrtStatus* EnsureMublasHandle(mublasHandle_t* handle) {
  if (g_handle == nullptr) {
    mublasStatus status = mublasCreate(&g_handle);
    if (status != MUBLAS_STATUS_SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasCreate failed");
    }
  }
  *handle = g_handle;
  return nullptr;
}

bool ResolveTF32Enabled() {
  const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
  return tf32_env != nullptr && std::atoi(tf32_env) != 0;
}

OrtStatus* EnsureMudnnHandle(::musa::dnn::Handle** handle) {
  if (!g_mudnn_handle) {
    g_mudnn_handle = std::make_unique<::musa::dnn::Handle>();
    auto status = g_mudnn_handle->SetAllowTF32(ResolveTF32Enabled());
    if (status != ::musa::dnn::Status::SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        "mudnn Handle SetAllowTF32 failed");
    }
  }
  *handle = g_mudnn_handle.get();
  return nullptr;
}

bool SetMudnnFloatTensor(::musa::dnn::Tensor& tensor, const void* data,
                         const std::vector<int64_t>& shape) {
  if (tensor.SetAddr(data) != ::musa::dnn::Status::SUCCESS) return false;
  if (tensor.SetType(::musa::dnn::Tensor::Type::FLOAT) !=
      ::musa::dnn::Status::SUCCESS)
    return false;
  if (tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW) !=
      ::musa::dnn::Status::SUCCESS)
    return false;
  return tensor.SetNdInfo(static_cast<int64_t>(shape.size()), shape.data()) ==
         ::musa::dnn::Status::SUCCESS;
}

bool TryMudnnMatMul(float* y_data, const float* a_data, const float* b_data,
                    const std::vector<int64_t>& a_shape,
                    const std::vector<int64_t>& b_shape,
                    const std::vector<int64_t>& y_shape) {
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
      !SetMudnnFloatTensor(y_tensor, y_data, y_shape)) {
    return false;
  }

  ::musa::dnn::MatMul matmul;
  if (matmul.SetTranspose(false, false) != ::musa::dnn::Status::SUCCESS)
    return false;
  if (matmul.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR) !=
      ::musa::dnn::Status::SUCCESS)
    return false;
  if (matmul.SetAlpha(1.0) != ::musa::dnn::Status::SUCCESS) return false;
  if (matmul.SetBeta(0.0) != ::musa::dnn::Status::SUCCESS) return false;
  return matmul.Run(*handle, y_tensor, a_tensor, b_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

int64_t NumElements(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

bool IsGpuMemory(const OrtMemoryInfo* memory_info) {
  const OrtMemoryDevice* device =
      Ort::GetEpApi().MemoryInfo_GetMemoryDevice(memory_info);
  return Ort::GetEpApi().MemoryDevice_GetDeviceType(device) ==
         OrtMemoryInfoDeviceType_GPU;
}

std::vector<int64_t> Strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<int64_t> Coordinates(int64_t linear,
                                 const std::vector<int64_t>& shape) {
  std::vector<int64_t> coord(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    int64_t dim = shape[static_cast<size_t>(i)];
    coord[static_cast<size_t>(i)] = dim == 0 ? 0 : linear % dim;
    linear = dim == 0 ? 0 : linear / dim;
  }
  return coord;
}

int64_t Offset(const std::vector<int64_t>& coord,
               const std::vector<int64_t>& strides) {
  int64_t off = 0;
  for (size_t i = 0; i < coord.size(); ++i) {
    off += coord[i] * strides[i];
  }
  return off;
}

std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a,
                                    const std::vector<int64_t>& b) {
  size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      throw std::runtime_error("MatMul batch broadcast shape mismatch");
    }
    out[i] = std::max(da, db);
  }
  return out;
}

std::vector<int64_t> PrefixShape(const std::vector<int64_t>& shape,
                                 size_t trailing_dims) {
  if (shape.size() < trailing_dims) {
    return {};
  }
  return std::vector<int64_t>(
      shape.begin(), shape.end() - static_cast<int64_t>(trailing_dims));
}

bool SameShape(const std::vector<int64_t>& lhs,
               const std::vector<int64_t>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool IsScalarBatch(const std::vector<int64_t>& shape) {
  return shape.empty() || NumElements(shape) == 1;
}

bool TryGetLinearBatchStride(const std::vector<int64_t>& input_batch,
                             const std::vector<int64_t>& output_batch,
                             int64_t matrix_elements, long long int& stride) {
  if (SameShape(input_batch, output_batch)) {
    stride = static_cast<long long int>(matrix_elements);
    return true;
  }
  if (IsScalarBatch(input_batch)) {
    stride = 0;
    return true;
  }
  return false;
}

std::vector<int64_t> BroadcastBatchCoord(const std::vector<int64_t>& out_coord,
                                         const std::vector<int64_t>& out_shape,
                                         const std::vector<int64_t>& in_shape) {
  std::vector<int64_t> coord(in_shape.size(), 0);
  size_t out_rank = out_shape.size();
  size_t in_rank = in_shape.size();
  for (size_t i = 0; i < in_rank; ++i) {
    size_t out_i = out_rank - in_rank + i;
    coord[i] = in_shape[i] == 1 ? 0 : out_coord[out_i];
  }
  return coord;
}

OrtStatus* ComputeMusaMatMulDeviceImpl(const float* a_data, const float* b_data,
                                       float* y_data,
                                       const std::vector<int64_t>& a_shape,
                                       const std::vector<int64_t>& b_shape,
                                       const std::vector<int64_t>& y_shape) {
  if (a_data == nullptr || b_data == nullptr || y_data == nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul device pointers must be non-null.");
  }
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul requires rank >= 2 tensors.");
  }

  int64_t m64 = a_shape[a_shape.size() - 2];
  int64_t k64 = a_shape[a_shape.size() - 1];
  int64_t kb64 = b_shape[b_shape.size() - 2];
  int64_t n64 = b_shape[b_shape.size() - 1];
  if (k64 != kb64) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul input shapes are incompatible.");
  }
  if (m64 <= 0 || k64 <= 0 || n64 <= 0) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MUSA MatMul requires positive static dimensions.");
  }
  if (m64 > INT32_MAX || k64 > INT32_MAX || n64 > INT32_MAX) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MUSA MatMul dimensions exceed int32 mublas limits.");
  }

  std::vector<int64_t> a_batch = PrefixShape(a_shape, 2);
  std::vector<int64_t> b_batch = PrefixShape(b_shape, 2);
  std::vector<int64_t> batch_shape = BroadcastShape(a_batch, b_batch);
  std::vector<int64_t> expected_y_shape = batch_shape;
  expected_y_shape.push_back(m64);
  expected_y_shape.push_back(n64);
  if (!SameShape(y_shape, expected_y_shape)) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "MUSA MatMul output shape mismatch.");
  }

  if (a_shape.size() == 2 && b_shape.size() == 2 &&
      m64 * k64 * n64 >= 1000000 &&
      TryMudnnMatMul(y_data, a_data, b_data, a_shape, b_shape, y_shape)) {
    return nullptr;
  }

  auto a_strides = Strides(a_shape);
  auto b_strides = Strides(b_shape);
  auto y_strides = Strides(y_shape);
  int64_t batch_total = NumElements(batch_shape);

  long long int stride_a = 0;
  long long int stride_b = 0;
  const bool can_use_strided_batched =
      batch_total > 1 && batch_total <= INT32_MAX &&
      TryGetLinearBatchStride(a_batch, batch_shape, m64 * k64, stride_a) &&
      TryGetLinearBatchStride(b_batch, batch_shape, k64 * n64, stride_b);
  if (can_use_strided_batched) {
    mublasHandle_t handle = nullptr;
    RETURN_IF_ERROR(EnsureMublasHandle(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m = static_cast<int>(m64);
    const int k = static_cast<int>(k64);
    const int n = static_cast<int>(n64);
    const long long int stride_y = static_cast<long long int>(m64 * n64);

    mublasStatus status = mublasSgemmStridedBatched(
        handle, MUBLAS_OP_N, MUBLAS_OP_N, n, m, k, &alpha, b_data, n, stride_b,
        a_data, k, stride_a, &beta, y_data, n, stride_y,
        static_cast<int>(batch_total));
    if (status == MUBLAS_STATUS_SUCCESS) {
      return nullptr;
    }
    // Fall through to the generic path if muBLAS rejects a shape/stride
    // combination.
  }

  if (batch_total > 1 && y_shape.size() <= kMusaMaxBroadcastRank &&
      m64 <= 128 && n64 <= 128 && k64 <= 128) {
    MusaBatchedMatMulParams params{};
    params.output_rank = static_cast<int32_t>(y_shape.size());
    params.batch_rank = static_cast<int32_t>(batch_shape.size());
    params.total_elements = NumElements(y_shape);
    params.m = m64;
    params.n = n64;
    params.k = k64;
    for (size_t dim = 0; dim < y_shape.size(); ++dim) {
      params.output_dims[dim] = y_shape[dim];
      params.output_strides[dim] = y_strides[dim];
    }
    const size_t a_batch_rank = a_shape.size() - 2;
    const size_t b_batch_rank = b_shape.size() - 2;
    const size_t a_batch_offset = batch_shape.size() - a_batch_rank;
    const size_t b_batch_offset = batch_shape.size() - b_batch_rank;
    for (size_t dim = 0; dim < batch_shape.size(); ++dim) {
      if (dim < a_batch_offset) {
        params.a_batch_strides[dim] = 0;
      } else {
        const size_t a_dim = dim - a_batch_offset;
        params.a_batch_strides[dim] =
            a_shape[a_dim] == 1 ? 0 : a_strides[a_dim];
      }
      if (dim < b_batch_offset) {
        params.b_batch_strides[dim] = 0;
      } else {
        const size_t b_dim = dim - b_batch_offset;
        params.b_batch_strides[dim] =
            b_shape[b_dim] == 1 ? 0 : b_strides[b_dim];
      }
    }
    params.a_row_stride = a_strides[a_shape.size() - 2];
    params.a_col_stride = a_strides[a_shape.size() - 1];
    params.b_row_stride = b_strides[b_shape.size() - 2];
    params.b_col_stride = b_strides[b_shape.size() - 1];
    musaError_t status = LaunchMusaBatchedMatMulFloatKernel(
        a_data, b_data, y_data, params, nullptr);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
    return nullptr;
  }

  mublasHandle_t handle = nullptr;
  RETURN_IF_ERROR(EnsureMublasHandle(&handle));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int m = static_cast<int>(m64);
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);

  for (int64_t batch_idx = 0; batch_idx < batch_total; ++batch_idx) {
    std::vector<int64_t> batch_coord = Coordinates(batch_idx, batch_shape);
    std::vector<int64_t> a_batch_coord =
        BroadcastBatchCoord(batch_coord, batch_shape, a_batch);
    std::vector<int64_t> b_batch_coord =
        BroadcastBatchCoord(batch_coord, batch_shape, b_batch);
    std::vector<int64_t> y_batch_coord = batch_coord;
    a_batch_coord.push_back(0);
    a_batch_coord.push_back(0);
    b_batch_coord.push_back(0);
    b_batch_coord.push_back(0);
    y_batch_coord.push_back(0);
    y_batch_coord.push_back(0);
    const float* a_ptr = a_data + Offset(a_batch_coord, a_strides);
    const float* b_ptr = b_data + Offset(b_batch_coord, b_strides);
    float* y_ptr = y_data + Offset(y_batch_coord, y_strides);
    mublasStatus status =
        mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N, n, m, k, &alpha, b_ptr, n,
                    a_ptr, k, &beta, y_ptr, n);
    if (status != MUBLAS_STATUS_SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasSgemm failed");
    }
  }

  return nullptr;
}

OrtStatus* ComputeMublasBatchedMatMul(Ort::KernelContext& kernel_context,
                                      Ort::ConstValue a, Ort::ConstValue b,
                                      const std::vector<int64_t>& a_shape,
                                      const std::vector<int64_t>& b_shape) {
  if (!IsGpuMemory(a.GetTensorMemoryInfo()) ||
      !IsGpuMemory(b.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "MatMul requires MUSA inputs");
  }
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul requires rank >= 2 tensors.");
  }

  const int64_t m64 = a_shape[a_shape.size() - 2];
  const int64_t k64 = a_shape[a_shape.size() - 1];
  const int64_t kb64 = b_shape[b_shape.size() - 2];
  const int64_t n64 = b_shape[b_shape.size() - 1];
  if (k64 != kb64) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul input shapes are incompatible.");
  }

  std::vector<int64_t> a_batch = PrefixShape(a_shape, 2);
  std::vector<int64_t> b_batch = PrefixShape(b_shape, 2);
  std::vector<int64_t> batch_shape = BroadcastShape(a_batch, b_batch);
  std::vector<int64_t> y_shape = batch_shape;
  y_shape.push_back(m64);
  y_shape.push_back(n64);
  Ort::UnownedValue y = kernel_context.GetOutput(0, y_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "MatMul requires MUSA output");
  }

  return ComputeMusaMatMulDeviceImpl(
      a.GetTensorData<float>(), b.GetTensorData<float>(),
      y.GetTensorMutableData<float>(), a_shape, b_shape, y_shape);
}
}  // namespace

OrtStatus* ComputeMusaMatMulDevice(const float* a_data, const float* b_data,
                                   float* y_data,
                                   const std::vector<int64_t>& a_shape,
                                   const std::vector<int64_t>& b_shape,
                                   const std::vector<int64_t>& y_shape) {
  return ComputeMusaMatMulDeviceImpl(a_data, b_data, y_data, a_shape, b_shape,
                                     y_shape);
}

MatMul::MatMul(const OrtKernelInfo* info, void* /*state*/, PrivateTag)
    : kernel_base{}, info_{info} {
  kernel_base.ort_version_supported = ORT_API_VERSION;
  kernel_base.Compute = ComputeImpl;
  kernel_base.Release = ReleaseImpl;
}

OrtStatus* MatMul::CreateKernelImpl(const OrtKernelInfo* info, void* state,
                                    OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto matmul = std::make_unique<MatMul>(info, state, PrivateTag{});
  kernel = reinterpret_cast<OrtKernelImpl*>(matmul.release());
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL MatMul::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* matmul = reinterpret_cast<MatMul*>(this_ptr);
  static_cast<void>(matmul->info_);

  Ort::KernelContext kernel_context(kernel_ctx);

  Ort::ConstValue a = kernel_context.GetInput(0);
  Ort::ConstValue b = kernel_context.GetInput(1);
  auto a_shape_info = a.GetTensorTypeAndShapeInfo();
  auto b_shape_info = b.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> a_shape = a_shape_info.GetShape();
  std::vector<int64_t> b_shape = b_shape_info.GetShape();

  return ComputeMublasBatchedMatMul(kernel_context, a, b, a_shape, b_shape);
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL MatMul::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete reinterpret_cast<MatMul*>(this_ptr);
}
