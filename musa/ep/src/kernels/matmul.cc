#include "matmul.h"

#include "runtime/musa_runtime.h"
#include <mublas.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    MatMul)

namespace {
thread_local mublasHandle_t g_handle = nullptr;

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

int64_t NumElements(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

bool IsGpuMemory(const OrtMemoryInfo* memory_info) {
  const OrtMemoryDevice* device = Ort::GetEpApi().MemoryInfo_GetMemoryDevice(memory_info);
  return Ort::GetEpApi().MemoryDevice_GetDeviceType(device) == OrtMemoryInfoDeviceType_GPU;
}

OrtStatus* CopyFloatToHost(Ort::ConstValue value, std::vector<float>& data) {
  size_t num_bytes = value.GetTensorSizeInBytes();
  data.resize(num_bytes / sizeof(float));
  if (num_bytes == 0) {
    return nullptr;
  }
  const void* src = value.GetTensorRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    musaError_t status = musaMemcpy(data.data(), src, num_bytes, musaMemcpyDeviceToHost);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(data.data(), src, num_bytes);
  }
  return nullptr;
}

OrtStatus* CopyFloatFromHost(Ort::UnownedValue value, const std::vector<float>& data) {
  size_t num_bytes = data.size() * sizeof(float);
  if (num_bytes == 0) {
    return nullptr;
  }
  void* dst = value.GetTensorMutableRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    musaError_t status = musaMemcpy(dst, data.data(), num_bytes, musaMemcpyHostToDevice);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(dst, data.data(), num_bytes);
  }
  return nullptr;
}

std::vector<int64_t> Strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<int64_t> Coordinates(int64_t linear, const std::vector<int64_t>& shape) {
  std::vector<int64_t> coord(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    int64_t dim = shape[static_cast<size_t>(i)];
    coord[static_cast<size_t>(i)] = dim == 0 ? 0 : linear % dim;
    linear = dim == 0 ? 0 : linear / dim;
  }
  return coord;
}

int64_t Offset(const std::vector<int64_t>& coord, const std::vector<int64_t>& strides) {
  int64_t off = 0;
  for (size_t i = 0; i < coord.size(); ++i) {
    off += coord[i] * strides[i];
  }
  return off;
}

std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
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

std::vector<int64_t> PrefixShape(const std::vector<int64_t>& shape, size_t trailing_dims) {
  if (shape.size() < trailing_dims) {
    return {};
  }
  return std::vector<int64_t>(shape.begin(), shape.end() - static_cast<int64_t>(trailing_dims));
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

OrtStatus* ComputeHostMatMul(Ort::KernelContext& kernel_context, Ort::ConstValue a, Ort::ConstValue b,
                             const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape) {
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul requires rank >= 2 tensors.");
  }
  int64_t m = a_shape[a_shape.size() - 2];
  int64_t k = a_shape[a_shape.size() - 1];
  int64_t kb = b_shape[b_shape.size() - 2];
  int64_t n = b_shape[b_shape.size() - 1];
  if (k != kb) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul input shapes are incompatible.");
  }

  std::vector<int64_t> a_batch = PrefixShape(a_shape, 2);
  std::vector<int64_t> b_batch = PrefixShape(b_shape, 2);
  std::vector<int64_t> batch_shape = BroadcastShape(a_batch, b_batch);
  std::vector<int64_t> y_shape = batch_shape;
  y_shape.push_back(m);
  y_shape.push_back(n);

  std::vector<float> a_host;
  std::vector<float> b_host;
  RETURN_IF_ERROR(CopyFloatToHost(a, a_host));
  RETURN_IF_ERROR(CopyFloatToHost(b, b_host));
  std::vector<float> y_host(static_cast<size_t>(NumElements(y_shape)), 0.0f);

  auto a_strides = Strides(a_shape);
  auto b_strides = Strides(b_shape);
  auto y_strides = Strides(y_shape);
  int64_t batch_total = NumElements(batch_shape);
  for (int64_t batch_idx = 0; batch_idx < batch_total; ++batch_idx) {
    std::vector<int64_t> batch_coord = Coordinates(batch_idx, batch_shape);
    std::vector<int64_t> a_batch_coord = BroadcastBatchCoord(batch_coord, batch_shape, a_batch);
    std::vector<int64_t> b_batch_coord = BroadcastBatchCoord(batch_coord, batch_shape, b_batch);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int64_t kk = 0; kk < k; ++kk) {
          std::vector<int64_t> a_coord = a_batch_coord;
          std::vector<int64_t> b_coord = b_batch_coord;
          a_coord.push_back(row);
          a_coord.push_back(kk);
          b_coord.push_back(kk);
          b_coord.push_back(col);
          sum += a_host[static_cast<size_t>(Offset(a_coord, a_strides))] *
                 b_host[static_cast<size_t>(Offset(b_coord, b_strides))];
        }
        std::vector<int64_t> y_coord = batch_coord;
        y_coord.push_back(row);
        y_coord.push_back(col);
        y_host[static_cast<size_t>(Offset(y_coord, y_strides))] = sum;
      }
    }
  }

  Ort::UnownedValue y = kernel_context.GetOutput(0, y_shape);
  return CopyFloatFromHost(y, y_host);
}

OrtStatus* ComputeMublasBatchedMatMul(Ort::KernelContext& kernel_context, Ort::ConstValue a, Ort::ConstValue b,
                                      const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape) {
  if (!IsGpuMemory(a.GetTensorMemoryInfo()) || !IsGpuMemory(b.GetTensorMemoryInfo())) {
    return ComputeHostMatMul(kernel_context, a, b, a_shape, b_shape);
  }
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul requires rank >= 2 tensors.");
  }
  int64_t m64 = a_shape[a_shape.size() - 2];
  int64_t k64 = a_shape[a_shape.size() - 1];
  int64_t kb64 = b_shape[b_shape.size() - 2];
  int64_t n64 = b_shape[b_shape.size() - 1];
  if (k64 != kb64) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul input shapes are incompatible.");
  }
  if (m64 > INT32_MAX || k64 > INT32_MAX || n64 > INT32_MAX) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul dimensions exceed int32 mublas limits.");
  }

  std::vector<int64_t> a_batch = PrefixShape(a_shape, 2);
  std::vector<int64_t> b_batch = PrefixShape(b_shape, 2);
  std::vector<int64_t> batch_shape = BroadcastShape(a_batch, b_batch);
  std::vector<int64_t> y_shape = batch_shape;
  y_shape.push_back(m64);
  y_shape.push_back(n64);
  Ort::UnownedValue y = kernel_context.GetOutput(0, y_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return ComputeHostMatMul(kernel_context, a, b, a_shape, b_shape);
  }

  const float* a_data = a.GetTensorData<float>();
  const float* b_data = b.GetTensorData<float>();
  float* y_data = y.GetTensorMutableData<float>();
  auto a_strides = Strides(a_shape);
  auto b_strides = Strides(b_shape);
  auto y_strides = Strides(y_shape);

  mublasHandle_t handle = nullptr;
  RETURN_IF_ERROR(EnsureMublasHandle(&handle));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int m = static_cast<int>(m64);
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);

  int64_t batch_total = NumElements(batch_shape);
  for (int64_t batch_idx = 0; batch_idx < batch_total; ++batch_idx) {
    std::vector<int64_t> batch_coord = Coordinates(batch_idx, batch_shape);
    std::vector<int64_t> a_batch_coord = BroadcastBatchCoord(batch_coord, batch_shape, a_batch);
    std::vector<int64_t> b_batch_coord = BroadcastBatchCoord(batch_coord, batch_shape, b_batch);
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
    mublasStatus status = mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                                      n, m, k,
                                      &alpha,
                                      b_ptr, n,
                                      a_ptr, k,
                                      &beta,
                                      y_ptr, n);
    if (status != MUBLAS_STATUS_SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasSgemm failed");
    }
  }

  musaError_t sync_status = musaDeviceSynchronize();
  if (sync_status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(sync_status));
  }
  return nullptr;
}
}  // namespace

MatMul::MatMul(const OrtKernelInfo* info, void* /*state*/, PrivateTag)
    : kernel_base{}, info_{info} {
  kernel_base.ort_version_supported = ORT_API_VERSION;
  kernel_base.Compute = ComputeImpl;
  kernel_base.Release = ReleaseImpl;
}

OrtStatus* MatMul::CreateKernelImpl(const OrtKernelInfo* info, void* state, OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto matmul = std::make_unique<MatMul>(info, state, PrivateTag{});
  kernel = reinterpret_cast<OrtKernelImpl*>(matmul.release());
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL MatMul::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
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

  if (a_shape.size() != 2 || b_shape.size() != 2) {
    return ComputeMublasBatchedMatMul(kernel_context, a, b, a_shape, b_shape);
  }
  if (a_shape[1] != b_shape[0]) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul input shapes are incompatible.");
  }
  if (a_shape[0] <= 0 || a_shape[1] <= 0 || b_shape[1] <= 0) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul skeleton requires positive static dimensions.");
  }

  const int64_t m64 = a_shape[0];
  const int64_t k64 = a_shape[1];
  const int64_t n64 = b_shape[1];
  if (m64 > INT32_MAX || k64 > INT32_MAX || n64 > INT32_MAX) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "MUSA MatMul dimensions exceed int32 mublas limits.");
  }
  const int m = static_cast<int>(m64);
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);

  std::vector<int64_t> y_shape{m64, n64};
  Ort::UnownedValue y = kernel_context.GetOutput(0, y_shape);

  const float* a_data = a.GetTensorData<float>();
  const float* b_data = b.GetTensorData<float>();
  float* y_data = y.GetTensorMutableData<float>();

  mublasHandle_t handle = nullptr;
  RETURN_IF_ERROR(EnsureMublasHandle(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // muBLAS uses column-major semantics. For row-major ONNX tensors:
  // C(m,n) = A(m,k) * B(k,n) is computed as C^T(n,m) = B^T(n,k) * A^T(k,m).
  mublasStatus status = mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    b_data, n,
                                    a_data, k,
                                    &beta,
                                    y_data, n);
  if (status != MUBLAS_STATUS_SUCCESS) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasSgemm failed");
  }

  musaError_t sync_status = musaDeviceSynchronize();
  if (sync_status != musaSuccess) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(sync_status));
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL MatMul::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete reinterpret_cast<MatMul*>(this_ptr);
}
