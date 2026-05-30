#include "basic_ops.h"

#include <mublas.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <sstream>
#include <unordered_set>

#include "runtime/musa_runtime.h"

namespace {

thread_local mublasHandle_t g_basic_mublas_handle = nullptr;

OrtStatus* EnsureBasicMublasHandle(mublasHandle_t* handle) {
  if (g_basic_mublas_handle == nullptr) {
    mublasStatus status = mublasCreate(&g_basic_mublas_handle);
    if (status != MUBLAS_STATUS_SUCCESS) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "mublasCreate failed");
    }
  }
  *handle = g_basic_mublas_handle;
  return nullptr;
}

constexpr int64_t kFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr int64_t kInt32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
constexpr int64_t kInt64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
constexpr int64_t kBool = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;

std::vector<const OrtDataType*> AllTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

std::vector<const OrtDataType*> FloatTensorTypes() {
  return {GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)};
}

std::vector<const OrtDataType*> IntTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

template <typename T>
T AttrOrDefault(Ort::ConstKernelInfo& info, const char* name, T default_value) {
  try {
    return info.GetAttribute<T>(name);
  } catch (...) {
    return default_value;
  }
}

std::vector<int64_t> AttrsOrEmpty(Ort::ConstKernelInfo& info,
                                  const char* name) {
  try {
    return info.GetAttributes<int64_t>(name);
  } catch (...) {
    return {};
  }
}

bool IsGpuMemory(const OrtMemoryInfo* memory_info) {
  const OrtMemoryDevice* device =
      Ort::GetEpApi().MemoryInfo_GetMemoryDevice(memory_info);
  return Ort::GetEpApi().MemoryDevice_GetDeviceType(device) ==
         OrtMemoryInfoDeviceType_GPU;
}

OrtStatus* CopyToHost(Ort::ConstValue value, std::vector<uint8_t>& bytes) {
  size_t num_bytes = value.GetTensorSizeInBytes();
  bytes.resize(num_bytes);
  if (num_bytes == 0) {
    return nullptr;
  }

  const void* src = value.GetTensorRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    musaError_t status =
        musaMemcpy(bytes.data(), src, num_bytes, musaMemcpyDeviceToHost);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(bytes.data(), src, num_bytes);
  }

  return nullptr;
}

OrtStatus* CopyFromHost(Ort::UnownedValue value, const void* src,
                        size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }

  void* dst = value.GetTensorMutableRawData();
  if (IsGpuMemory(value.GetTensorMemoryInfo())) {
    musaError_t status =
        musaMemcpy(dst, src, num_bytes, musaMemcpyHostToDevice);
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
  } else {
    std::memcpy(dst, src, num_bytes);
  }

  return nullptr;
}

size_t ElementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return 1;
    default:
      return 0;
  }
}

int64_t NumElements(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

int64_t NormalizeAxis(int64_t axis, size_t rank) {
  int64_t r = static_cast<int64_t>(rank);
  return axis < 0 ? axis + r : axis;
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
      throw std::runtime_error("broadcast shape mismatch");
    }
    out[i] = std::max(da, db);
  }
  return out;
}

int64_t BroadcastOffset(const std::vector<int64_t>& out_coord,
                        const std::vector<int64_t>& in_shape,
                        const std::vector<int64_t>& in_strides) {
  size_t rank = out_coord.size();
  size_t in_rank = in_shape.size();
  int64_t off = 0;
  for (size_t i = 0; i < in_rank; ++i) {
    size_t out_i = rank - in_rank + i;
    int64_t c = in_shape[i] == 1 ? 0 : out_coord[out_i];
    off += c * in_strides[i];
  }
  return off;
}

template <typename T>
std::span<const T> Span(const std::vector<uint8_t>& bytes) {
  return std::span<const T>(reinterpret_cast<const T*>(bytes.data()),
                            bytes.size() / sizeof(T));
}

template <typename T>
std::vector<T> ReadTyped(Ort::ConstValue value) {
  std::vector<uint8_t> bytes;
  Ort::ThrowOnError(CopyToHost(value, bytes));
  std::vector<T> out(bytes.size() / sizeof(T));
  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }
  return out;
}

template <typename T>
OrtStatus* WriteTyped(Ort::UnownedValue value, const std::vector<T>& data) {
  return CopyFromHost(value, data.data(), data.size() * sizeof(T));
}

template <typename T, typename Fn>
OrtStatus* BinaryCompute(Ort::KernelContext& ctx,
                         const std::vector<int64_t>& shape0,
                         const std::vector<int64_t>& shape1, Fn fn) {
  std::vector<T> a = ReadTyped<T>(ctx.GetInput(0));
  std::vector<T> b = ReadTyped<T>(ctx.GetInput(1));
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
  return WriteTyped<T>(y, out);
}

template <typename T, typename Fn>
OrtStatus* UnaryCompute(Ort::KernelContext& ctx,
                        const std::vector<int64_t>& shape, Fn fn) {
  std::vector<T> x = ReadTyped<T>(ctx.GetInput(0));
  std::vector<T> y_data(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    y_data[i] = fn(x[i]);
  }
  Ort::UnownedValue y = ctx.GetOutput(0, shape);
  return WriteTyped<T>(y, y_data);
}

std::vector<int64_t> ReadIntTensor(Ort::KernelContext& ctx, size_t index) {
  Ort::ConstValue value = ctx.GetInput(index);
  auto info = value.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ReadTyped<int64_t>(value);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> vals = ReadTyped<int32_t>(value);
    return std::vector<int64_t>(vals.begin(), vals.end());
  }
  throw std::runtime_error("expected int32/int64 tensor");
}

std::set<int64_t> AxesSet(std::vector<int64_t> axes, size_t rank) {
  std::set<int64_t> out;
  if (axes.empty()) {
    for (size_t i = 0; i < rank; ++i) out.insert(static_cast<int64_t>(i));
    return out;
  }
  for (int64_t axis : axes) {
    out.insert(NormalizeAxis(axis, rank));
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

void ApplyActivation(std::vector<float>& values, const std::string& activation,
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

OrtStatus* GemmCompute(Ort::KernelContext& ctx, bool trans_a, bool trans_b,
                       float alpha, float beta, const std::string& activation,
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
      RETURN_IF_ERROR(EnsureBasicMublasHandle(&handle));
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

OrtStatus* FusedMatMulCompute(Ort::KernelContext& ctx, bool trans_a,
                              bool trans_b, bool trans_batch_a,
                              bool trans_batch_b, float alpha) {
  if (trans_batch_a || trans_batch_b) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "FusedMatMul transBatch is not implemented");
  }
  auto a_info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto b_info = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  if (a_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "FusedMatMul only supports float tensors");
  }

  std::vector<int64_t> a_shape = a_info.GetShape();
  std::vector<int64_t> b_shape = b_info.GetShape();
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "FusedMatMul requires rank >= 2 inputs");
  }

  int64_t m =
      trans_a ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
  int64_t k =
      trans_a ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
  int64_t kb =
      trans_b ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
  int64_t n =
      trans_b ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  if (k != kb) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "FusedMatMul K dimension mismatch");
  }

  std::vector<int64_t> a_batch = PrefixShape(a_shape, 2);
  std::vector<int64_t> b_batch = PrefixShape(b_shape, 2);
  std::vector<int64_t> batch_shape = BroadcastShape(a_batch, b_batch);
  std::vector<int64_t> out_shape = batch_shape;
  out_shape.push_back(m);
  out_shape.push_back(n);

  std::vector<float> a = ReadTyped<float>(ctx.GetInput(0));
  std::vector<float> b = ReadTyped<float>(ctx.GetInput(1));
  std::vector<float> out(static_cast<size_t>(NumElements(out_shape)), 0.0f);
  auto a_strides = Strides(a_shape);
  auto b_strides = Strides(b_shape);
  auto out_strides = Strides(out_shape);
  int64_t batch_total = NumElements(batch_shape);

  for (int64_t batch_idx = 0; batch_idx < batch_total; ++batch_idx) {
    std::vector<int64_t> batch_coord = Coordinates(batch_idx, batch_shape);
    std::vector<int64_t> a_batch_coord =
        BroadcastBatchCoord(batch_coord, batch_shape, a_batch);
    std::vector<int64_t> b_batch_coord =
        BroadcastBatchCoord(batch_coord, batch_shape, b_batch);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int64_t kk = 0; kk < k; ++kk) {
          std::vector<int64_t> a_coord = a_batch_coord;
          std::vector<int64_t> b_coord = b_batch_coord;
          if (trans_a) {
            a_coord.push_back(kk);
            a_coord.push_back(row);
          } else {
            a_coord.push_back(row);
            a_coord.push_back(kk);
          }
          if (trans_b) {
            b_coord.push_back(col);
            b_coord.push_back(kk);
          } else {
            b_coord.push_back(kk);
            b_coord.push_back(col);
          }
          sum += a[static_cast<size_t>(Offset(a_coord, a_strides))] *
                 b[static_cast<size_t>(Offset(b_coord, b_strides))];
        }
        std::vector<int64_t> out_coord = batch_coord;
        out_coord.push_back(row);
        out_coord.push_back(col);
        out[static_cast<size_t>(Offset(out_coord, out_strides))] = alpha * sum;
      }
    }
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<float>(y, out);
}

}  // namespace

BasicOp::BasicOp(const OrtKernelInfo* info, void* /*state*/, PrivateTag)
    : OrtKernelImpl{} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;

  Ort::ConstKernelInfo kernel_info(info);
  op_type_ = kernel_info.GetOperatorType();
  axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  keepdims_ = AttrOrDefault<int64_t>(kernel_info, "keepdims", 1);
  allowzero_ = AttrOrDefault<int64_t>(kernel_info, "allowzero", 0);
  to_ = AttrOrDefault<int64_t>(kernel_info, "to", 0);
  trans_a_ = AttrOrDefault<int64_t>(kernel_info, "transA", 0);
  trans_b_ = AttrOrDefault<int64_t>(kernel_info, "transB", 0);
  trans_batch_a_ = AttrOrDefault<int64_t>(kernel_info, "transBatchA", 0);
  trans_batch_b_ = AttrOrDefault<int64_t>(kernel_info, "transBatchB", 0);
  alpha_gemm_ = AttrOrDefault<float>(kernel_info, "alpha", 1.0f);
  beta_ = AttrOrDefault<float>(kernel_info, "beta", 1.0f);
  alpha_ = AttrOrDefault<float>(kernel_info, "alpha", 0.01f);
  alpha_ = AttrOrDefault<float>(kernel_info, "activation_alpha", alpha_);
  activation_ = AttrOrDefault<std::string>(kernel_info, "activation", "");
  axes_attr_ = AttrsOrEmpty(kernel_info, "axes");
  perm_attr_ = AttrsOrEmpty(kernel_info, "perm");
}

OrtStatus* BasicOp::CreateKernelImpl(const OrtKernelInfo* info, void* state,
                                     OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto k = std::make_unique<BasicOp>(info, state, PrivateTag{});
  kernel = k.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL BasicOp::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  BasicOp* k = static_cast<BasicOp*>(this_ptr);
  Ort::KernelContext ctx(kernel_ctx);
  return k->ComputeInternal(ctx);
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL BasicOp::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<BasicOp*>(this_ptr);
}

OrtStatus* BasicOp::ComputeInternal(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();

  if (op_type_ == "Gemm" || op_type_ == "FusedGemm") {
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Gemm only supports float tensors");
    }
    return GemmCompute(ctx, trans_a_ != 0, trans_b_ != 0, alpha_gemm_, beta_,
                       op_type_ == "FusedGemm" ? activation_ : std::string{},
                       alpha_);
  }

  if (op_type_ == "FusedMatMul") {
    return FusedMatMulCompute(ctx, trans_a_ != 0, trans_b_ != 0,
                              trans_batch_a_ != 0, trans_batch_b_ != 0,
                              alpha_gemm_);
  }

  if (op_type_ == "Add" || op_type_ == "Sub" || op_type_ == "Mul" ||
      op_type_ == "Div" || op_type_ == "Pow") {
    auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      if (op_type_ == "Add")
        return BinaryCompute<float>(ctx, shape0, shape1,
                                    [](float a, float b) { return a + b; });
      if (op_type_ == "Sub")
        return BinaryCompute<float>(ctx, shape0, shape1,
                                    [](float a, float b) { return a - b; });
      if (op_type_ == "Mul")
        return BinaryCompute<float>(ctx, shape0, shape1,
                                    [](float a, float b) { return a * b; });
      if (op_type_ == "Div")
        return BinaryCompute<float>(ctx, shape0, shape1,
                                    [](float a, float b) { return a / b; });
      return BinaryCompute<float>(
          ctx, shape0, shape1, [](float a, float b) { return std::pow(a, b); });
    }
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      if (op_type_ == "Add")
        return BinaryCompute<int64_t>(
            ctx, shape0, shape1, [](int64_t a, int64_t b) { return a + b; });
      if (op_type_ == "Sub")
        return BinaryCompute<int64_t>(
            ctx, shape0, shape1, [](int64_t a, int64_t b) { return a - b; });
      if (op_type_ == "Mul")
        return BinaryCompute<int64_t>(
            ctx, shape0, shape1, [](int64_t a, int64_t b) { return a * b; });
    }
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported binary op dtype");
  }

  if (op_type_ == "Relu" || op_type_ == "LeakyRelu" || op_type_ == "Sqrt" ||
      op_type_ == "Reciprocal" || op_type_ == "Neg" || op_type_ == "Log" ||
      op_type_ == "Tanh" || op_type_ == "Sigmoid") {
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "unsupported unary op dtype");
    }
    if (op_type_ == "Relu")
      return UnaryCompute<float>(ctx, shape0,
                                 [](float x) { return std::max(0.0f, x); });
    if (op_type_ == "LeakyRelu") {
      float alpha = alpha_;
      return UnaryCompute<float>(
          ctx, shape0, [alpha](float x) { return x >= 0.0f ? x : alpha * x; });
    }
    if (op_type_ == "Sqrt")
      return UnaryCompute<float>(ctx, shape0,
                                 [](float x) { return std::sqrt(x); });
    if (op_type_ == "Reciprocal")
      return UnaryCompute<float>(ctx, shape0, [](float x) { return 1.0f / x; });
    if (op_type_ == "Neg")
      return UnaryCompute<float>(ctx, shape0, [](float x) { return -x; });
    if (op_type_ == "Log")
      return UnaryCompute<float>(ctx, shape0,
                                 [](float x) { return std::log(x); });
    if (op_type_ == "Tanh")
      return UnaryCompute<float>(ctx, shape0,
                                 [](float x) { return std::tanh(x); });
    return UnaryCompute<float>(
        ctx, shape0, [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  }

  if (op_type_ == "Sum") {
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Sum only supports float");
    }
    std::vector<int64_t> out_shape = shape0;
    std::vector<float> out = ReadTyped<float>(input0);
    for (size_t idx = 1; idx < ctx.GetInputCount(); ++idx) {
      auto shape = ctx.GetInput(idx).GetTensorTypeAndShapeInfo().GetShape();
      out_shape = BroadcastShape(out_shape, shape);
      std::vector<float> lhs = out;
      std::vector<int64_t> lhs_shape = idx == 1 ? shape0 : out_shape;
      std::vector<float> rhs = ReadTyped<float>(ctx.GetInput(idx));
      std::vector<float> next(static_cast<size_t>(NumElements(out_shape)),
                              0.0f);
      auto ls = Strides(lhs_shape), rs = Strides(shape);
      for (int64_t i = 0; i < NumElements(out_shape); ++i) {
        auto coord = Coordinates(i, out_shape);
        next[static_cast<size_t>(i)] =
            lhs[static_cast<size_t>(BroadcastOffset(coord, lhs_shape, ls))] +
            rhs[static_cast<size_t>(BroadcastOffset(coord, shape, rs))];
      }
      out = std::move(next);
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return WriteTyped<float>(y, out);
  }

  if (op_type_ == "Shape") {
    int64_t rank = static_cast<int64_t>(shape0.size());
    int64_t start = 0;
    int64_t end = rank;
    // start/end attributes are uncommon in this model. Keep default full shape
    // for now.
    std::vector<int64_t> out(shape0.begin() + start, shape0.begin() + end);
    Ort::UnownedValue y = ctx.GetOutput(0, {static_cast<int64_t>(out.size())});
    return WriteTyped<int64_t>(y, out);
  }

  if (op_type_ == "Cast") {
    std::vector<uint8_t> in;
    RETURN_IF_ERROR(CopyToHost(input0, in));
    int64_t n = NumElements(shape0);
    Ort::UnownedValue y = ctx.GetOutput(0, shape0);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && to_ == kInt64) {
      auto x = Span<float>(in);
      std::vector<int64_t> out(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<int64_t>(x[static_cast<size_t>(i)]);
      return WriteTyped<int64_t>(y, out);
    }
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && to_ == kInt32) {
      auto x = Span<float>(in);
      std::vector<int32_t> out(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<int32_t>(x[static_cast<size_t>(i)]);
      return WriteTyped<int32_t>(y, out);
    }
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && to_ == kFloat) {
      auto x = Span<int64_t>(in);
      std::vector<float> out(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<float>(x[static_cast<size_t>(i)]);
      return WriteTyped<float>(y, out);
    }
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && to_ == kFloat) {
      auto x = Span<int32_t>(in);
      std::vector<float> out(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<float>(x[static_cast<size_t>(i)]);
      return WriteTyped<float>(y, out);
    }
    if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && to_ == kInt32) ||
        (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && to_ == kInt64)) {
      if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        auto x = Span<int64_t>(in);
        std::vector<int32_t> out(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
          out[static_cast<size_t>(i)] =
              static_cast<int32_t>(x[static_cast<size_t>(i)]);
        return WriteTyped<int32_t>(y, out);
      }
      auto x = Span<int32_t>(in);
      std::vector<int64_t> out(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<int64_t>(x[static_cast<size_t>(i)]);
      return WriteTyped<int64_t>(y, out);
    }
    return CopyFromHost(y, in.data(), in.size());
  }

  if (op_type_ == "Reshape") {
    std::vector<int64_t> requested = ReadIntTensor(ctx, 1);
    std::vector<int64_t> out_shape = requested;
    int64_t input_size = NumElements(shape0);
    int64_t known = 1;
    int infer_idx = -1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      if (out_shape[i] == 0 && !allowzero_) out_shape[i] = shape0[i];
      if (out_shape[i] == -1) {
        infer_idx = static_cast<int>(i);
      } else {
        known *= out_shape[i];
      }
    }
    if (infer_idx >= 0)
      out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
    std::vector<uint8_t> in;
    RETURN_IF_ERROR(CopyToHost(input0, in));
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return CopyFromHost(y, in.data(), in.size());
  }

  if (op_type_ == "Squeeze" || op_type_ == "Unsqueeze") {
    std::vector<int64_t> axes = axes_attr_;
    if (ctx.GetInputCount() > 1) axes = ReadIntTensor(ctx, 1);
    std::vector<uint8_t> in;
    RETURN_IF_ERROR(CopyToHost(input0, in));
    std::vector<int64_t> out_shape;
    if (op_type_ == "Unsqueeze") {
      int64_t out_rank = static_cast<int64_t>(shape0.size() + axes.size());
      std::set<int64_t> ax;
      for (int64_t a : axes) ax.insert(a < 0 ? a + out_rank : a);
      size_t src = 0;
      for (int64_t i = 0; i < out_rank; ++i) {
        out_shape.push_back(ax.count(i) ? 1 : shape0[src++]);
      }
    } else {
      std::set<int64_t> ax = AxesSet(axes, shape0.size());
      for (size_t i = 0; i < shape0.size(); ++i) {
        if (!ax.count(static_cast<int64_t>(i))) out_shape.push_back(shape0[i]);
      }
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return CopyFromHost(y, in.data(), in.size());
  }

  if (op_type_ == "Concat") {
    int64_t axis = NormalizeAxis(axis_, shape0.size());
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<uint8_t>> inputs;
    size_t elem_size = ElementSize(elem_type);
    std::vector<int64_t> out_shape = shape0;
    out_shape[static_cast<size_t>(axis)] = 0;
    for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
      auto v = ctx.GetInput(i);
      shapes.push_back(v.GetTensorTypeAndShapeInfo().GetShape());
      inputs.emplace_back();
      RETURN_IF_ERROR(CopyToHost(v, inputs.back()));
      out_shape[static_cast<size_t>(axis)] +=
          shapes.back()[static_cast<size_t>(axis)];
    }
    std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                             elem_size);
    auto out_strides = Strides(out_shape);
    std::vector<int64_t> axis_offsets;
    int64_t acc = 0;
    for (const auto& s : shapes) {
      axis_offsets.push_back(acc);
      acc += s[static_cast<size_t>(axis)];
    }
    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
      auto in_strides = Strides(shapes[input_idx]);
      int64_t total = NumElements(shapes[input_idx]);
      for (int64_t i = 0; i < total; ++i) {
        auto coord = Coordinates(i, shapes[input_idx]);
        auto out_coord = coord;
        out_coord[static_cast<size_t>(axis)] += axis_offsets[input_idx];
        std::memcpy(
            out.data() +
                static_cast<size_t>(Offset(out_coord, out_strides)) * elem_size,
            inputs[input_idx].data() + static_cast<size_t>(i) * elem_size,
            elem_size);
      }
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return CopyFromHost(y, out.data(), out.size());
  }

  if (op_type_ == "Transpose") {
    std::vector<int64_t> perm = perm_attr_;
    if (perm.empty()) {
      for (int64_t i = static_cast<int64_t>(shape0.size()) - 1; i >= 0; --i)
        perm.push_back(i);
    }
    std::vector<int64_t> out_shape;
    for (int64_t p : perm) out_shape.push_back(shape0[static_cast<size_t>(p)]);
    size_t elem_size = ElementSize(elem_type);
    std::vector<uint8_t> in;
    RETURN_IF_ERROR(CopyToHost(input0, in));
    std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                             elem_size);
    auto in_strides = Strides(shape0);
    for (int64_t i = 0; i < NumElements(out_shape); ++i) {
      auto out_coord = Coordinates(i, out_shape);
      std::vector<int64_t> in_coord(shape0.size());
      for (size_t j = 0; j < perm.size(); ++j)
        in_coord[static_cast<size_t>(perm[j])] = out_coord[j];
      std::memcpy(
          out.data() + static_cast<size_t>(i) * elem_size,
          in.data() +
              static_cast<size_t>(Offset(in_coord, in_strides)) * elem_size,
          elem_size);
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return CopyFromHost(y, out.data(), out.size());
  }

  if (op_type_ == "Gather") {
    int64_t axis = NormalizeAxis(axis_, shape0.size());
    std::vector<int64_t> indices = ReadIntTensor(ctx, 1);
    auto indices_shape = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> out_shape;
    out_shape.insert(out_shape.end(), shape0.begin(), shape0.begin() + axis);
    out_shape.insert(out_shape.end(), indices_shape.begin(),
                     indices_shape.end());
    out_shape.insert(out_shape.end(), shape0.begin() + axis + 1, shape0.end());
    size_t elem_size = ElementSize(elem_type);
    std::vector<uint8_t> in;
    RETURN_IF_ERROR(CopyToHost(input0, in));
    std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                             elem_size);
    auto in_strides = Strides(shape0);
    for (int64_t i = 0; i < NumElements(out_shape); ++i) {
      auto oc = Coordinates(i, out_shape);
      std::vector<int64_t> ic(shape0.size(), 0);
      for (int64_t d = 0; d < axis; ++d)
        ic[static_cast<size_t>(d)] = oc[static_cast<size_t>(d)];
      int64_t idx_offset = 0;
      auto idx_strides = Strides(indices_shape);
      for (size_t j = 0; j < indices_shape.size(); ++j)
        idx_offset += oc[static_cast<size_t>(axis) + j] * idx_strides[j];
      int64_t gather_idx = indices[static_cast<size_t>(idx_offset)];
      if (gather_idx < 0) gather_idx += shape0[static_cast<size_t>(axis)];
      ic[static_cast<size_t>(axis)] = gather_idx;
      for (size_t d = static_cast<size_t>(axis) + 1; d < shape0.size(); ++d) {
        ic[d] = oc[d - 1 + indices_shape.size()];
      }
      std::memcpy(
          out.data() + static_cast<size_t>(i) * elem_size,
          in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
          elem_size);
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return CopyFromHost(y, out.data(), out.size());
  }

  if (op_type_ == "Slice") {
    std::vector<int64_t> starts = ReadIntTensor(ctx, 1);
    std::vector<int64_t> ends = ReadIntTensor(ctx, 2);
    std::vector<int64_t> axes;
    std::vector<int64_t> steps(starts.size(), 1);
    if (ctx.GetInputCount() > 3) axes = ReadIntTensor(ctx, 3);
    if (ctx.GetInputCount() > 4) steps = ReadIntTensor(ctx, 4);
    if (axes.empty()) {
      axes.resize(starts.size());
      std::iota(axes.begin(), axes.end(), 0);
    }
    std::vector<int64_t> out_shape = shape0;
    std::vector<int64_t> norm_starts(shape0.size(), 0),
        norm_steps(shape0.size(), 1);
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t axis = NormalizeAxis(axes[i], shape0.size());
      int64_t dim = shape0[static_cast<size_t>(axis)];
      int64_t step = steps[i];
      if (step <= 0)
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED, "Slice negative step not implemented");
      int64_t start = starts[i] < 0 ? starts[i] + dim : starts[i];
      int64_t end = ends[i] < 0 ? ends[i] + dim : ends[i];
      start = std::max<int64_t>(0, std::min(start, dim));
      end = std::max<int64_t>(0, std::min(end, dim));
      norm_starts[static_cast<size_t>(axis)] = start;
      norm_steps[static_cast<size_t>(axis)] = step;
      out_shape[static_cast<size_t>(axis)] =
          std::max<int64_t>(0, (end - start + step - 1) / step);
    }
    size_t elem_size = ElementSize(elem_type);
    std::vector<uint8_t> in;
    RETURN_IF_ERROR(CopyToHost(input0, in));
    std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                             elem_size);
    auto in_strides = Strides(shape0);
    for (int64_t i = 0; i < NumElements(out_shape); ++i) {
      auto oc = Coordinates(i, out_shape);
      auto ic = oc;
      for (size_t d = 0; d < ic.size(); ++d)
        ic[d] = norm_starts[d] + oc[d] * norm_steps[d];
      std::memcpy(
          out.data() + static_cast<size_t>(i) * elem_size,
          in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
          elem_size);
    }
    Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
    return CopyFromHost(y, out.data(), out.size());
  }

  if (op_type_ == "Split") {
    int64_t axis = NormalizeAxis(axis_, shape0.size());
    std::vector<int64_t> splits;
    if (ctx.GetInputCount() > 1) {
      splits = ReadIntTensor(ctx, 1);
    } else {
      size_t count = ctx.GetOutputCount();
      splits.assign(count, shape0[static_cast<size_t>(axis)] /
                               static_cast<int64_t>(count));
    }
    size_t elem_size = ElementSize(elem_type);
    std::vector<uint8_t> in;
    RETURN_IF_ERROR(CopyToHost(input0, in));
    auto in_strides = Strides(shape0);
    int64_t axis_start = 0;
    for (size_t out_idx = 0; out_idx < splits.size(); ++out_idx) {
      std::vector<int64_t> out_shape = shape0;
      out_shape[static_cast<size_t>(axis)] = splits[out_idx];
      std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                               elem_size);
      for (int64_t i = 0; i < NumElements(out_shape); ++i) {
        auto oc = Coordinates(i, out_shape);
        auto ic = oc;
        ic[static_cast<size_t>(axis)] += axis_start;
        std::memcpy(
            out.data() + static_cast<size_t>(i) * elem_size,
            in.data() + static_cast<size_t>(Offset(ic, in_strides)) * elem_size,
            elem_size);
      }
      Ort::UnownedValue y = ctx.GetOutput(out_idx, out_shape);
      RETURN_IF_ERROR(CopyFromHost(y, out.data(), out.size()));
      axis_start += splits[out_idx];
    }
    return nullptr;
  }

  if (op_type_ == "ReduceProd" || op_type_ == "ReduceSum" ||
      op_type_ == "ReduceMean") {
    std::vector<int64_t> axes = axes_attr_;
    if (ctx.GetInputCount() > 1) axes = ReadIntTensor(ctx, 1);
    auto axes_set = AxesSet(axes, shape0.size());
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape0.size(); ++i) {
      if (axes_set.count(static_cast<int64_t>(i))) {
        if (keepdims_) out_shape.push_back(1);
      } else {
        out_shape.push_back(shape0[i]);
      }
    }
    if (out_shape.empty()) out_shape.push_back(1);
    auto out_strides = Strides(out_shape);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      std::vector<int64_t> x = ReadTyped<int64_t>(input0);
      std::vector<int64_t> out(static_cast<size_t>(NumElements(out_shape)),
                               op_type_ == "ReduceProd" ? 1 : 0);
      auto in_strides = Strides(shape0);
      for (int64_t i = 0; i < NumElements(shape0); ++i) {
        auto ic = Coordinates(i, shape0);
        std::vector<int64_t> oc;
        for (size_t d = 0; d < shape0.size(); ++d) {
          if (axes_set.count(static_cast<int64_t>(d))) {
            if (keepdims_) oc.push_back(0);
          } else {
            oc.push_back(ic[d]);
          }
        }
        if (oc.empty()) oc.push_back(0);
        int64_t oo = Offset(oc, out_strides);
        if (op_type_ == "ReduceProd")
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
                             op_type_ == "ReduceProd" ? 1.0f : 0.0f);
      std::vector<int64_t> counts(out.size(), 0);
      for (int64_t i = 0; i < NumElements(shape0); ++i) {
        auto ic = Coordinates(i, shape0);
        std::vector<int64_t> oc;
        for (size_t d = 0; d < shape0.size(); ++d) {
          if (axes_set.count(static_cast<int64_t>(d))) {
            if (keepdims_) oc.push_back(0);
          } else {
            oc.push_back(ic[d]);
          }
        }
        if (oc.empty()) oc.push_back(0);
        int64_t oo = Offset(oc, out_strides);
        if (op_type_ == "ReduceProd")
          out[static_cast<size_t>(oo)] *= x[static_cast<size_t>(i)];
        else
          out[static_cast<size_t>(oo)] += x[static_cast<size_t>(i)];
        counts[static_cast<size_t>(oo)]++;
      }
      if (op_type_ == "ReduceMean") {
        for (size_t i = 0; i < out.size(); ++i)
          out[i] /= static_cast<float>(counts[i]);
      }
      Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
      return WriteTyped<float>(y, out);
    }
  }

  if (op_type_ == "Softmax") {
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "Softmax only supports float");
    int64_t axis = NormalizeAxis(axis_, shape0.size());
    int64_t outer = 1, dim = shape0[static_cast<size_t>(axis)], inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= shape0[static_cast<size_t>(i)];
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape0.size(); ++i)
      inner *= shape0[i];
    std::vector<float> x = ReadTyped<float>(input0);
    std::vector<float> out(x.size());
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t in = 0; in < inner; ++in) {
        float max_v = -std::numeric_limits<float>::infinity();
        for (int64_t d = 0; d < dim; ++d)
          max_v = std::max(max_v,
                           x[static_cast<size_t>((o * dim + d) * inner + in)]);
        float sum = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
          float e = std::exp(
              x[static_cast<size_t>((o * dim + d) * inner + in)] - max_v);
          out[static_cast<size_t>((o * dim + d) * inner + in)] = e;
          sum += e;
        }
        for (int64_t d = 0; d < dim; ++d)
          out[static_cast<size_t>((o * dim + d) * inner + in)] /= sum;
      }
    }
    Ort::UnownedValue y = ctx.GetOutput(0, shape0);
    return WriteTyped<float>(y, out);
  }

  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    ("unsupported op: " + op_type_).c_str());
}

// Kernel definitions.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Add, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sub, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Div, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pow, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sum, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    FusedGemm, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    FusedMatMul, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Relu, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LeakyRelu, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sqrt, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reciprocal, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Neg, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Log, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tanh, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sigmoid, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T1", AllTensorTypes())
         .AddTypeConstraint("T2", AllTensorTypes())),
    BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Gather, kOnnxDomain, 13, 17,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", AllTensorTypes())
                                       .AddTypeConstraint("Tind",
                                                          IntTensorTypes())),
                                  BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Slice, kOnnxDomain, 13, 17,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", AllTensorTypes())
                                       .AddTypeConstraint("Tind",
                                                          IntTensorTypes())),
                                  BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceProd, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceSum, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllTensorTypes())), BasicOp)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ReduceMean, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    BasicOp)
