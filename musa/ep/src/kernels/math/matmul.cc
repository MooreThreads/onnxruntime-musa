#include "matmul.h"

#include <mublas.h>
#include <mudnncxx/mudnn.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "matmul_batched_kernels.h"
#include "shared_inc/blas_utils.h"

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    MatMul)

namespace {

struct EffectiveMatMulOperand {
  std::vector<int64_t> batch_shape;
  std::vector<int64_t> batch_strides;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t row_stride = 0;
  int64_t col_stride = 0;
};

struct MatMulShapeInfo {
  EffectiveMatMulOperand lhs;
  EffectiveMatMulOperand rhs;
  std::vector<int64_t> batch_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> compute_shape;
  bool lhs_vector = false;
  bool rhs_vector = false;
};

bool IsMatMulType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

int64_t NumElementsLocal(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    n *= dim;
  }
  return n;
}

std::vector<int64_t> StridesLocal(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

bool SameShape(const std::vector<int64_t>& lhs,
               const std::vector<int64_t>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

std::string ShapeToStringLocal(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::vector<int64_t> BroadcastShapeLocal(const std::vector<int64_t>& a,
                                         const std::vector<int64_t>& b) {
  size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      throw std::runtime_error("MatMul batch broadcast shape mismatch: lhs=" +
                               ShapeToStringLocal(a) + " rhs=" +
                               ShapeToStringLocal(b));
    }
    out[i] = da == 0 || db == 0 ? 0 : std::max(da, db);
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

bool IsScalarBatch(const std::vector<int64_t>& shape) {
  return shape.empty() || NumElementsLocal(shape) == 1;
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

void FillBroadcastBatchStrides(const std::vector<int64_t>& input_batch,
                               const std::vector<int64_t>& input_strides,
                               const std::vector<int64_t>& output_batch,
                               int64_t* output_strides) {
  const size_t output_rank = output_batch.size();
  const size_t input_rank = input_batch.size();
  const size_t input_offset = output_rank - input_rank;
  for (size_t dim = 0; dim < output_rank; ++dim) {
    if (dim < input_offset) {
      output_strides[dim] = 0;
      continue;
    }
    const size_t input_dim = dim - input_offset;
    output_strides[dim] =
        input_batch[input_dim] == 1 ? 0 : input_strides[input_dim];
  }
}

EffectiveMatMulOperand BuildOperand(const std::vector<int64_t>& shape,
                                    bool trans, bool trans_batch,
                                    bool is_left) {
  if (shape.empty()) {
    throw std::runtime_error("MatMul input rank must be >= 1");
  }
  auto strides = StridesLocal(shape);
  EffectiveMatMulOperand operand;

  if (shape.size() == 1) {
    operand.rows = is_left ? 1 : shape[0];
    operand.cols = is_left ? shape[0] : 1;
    operand.row_stride = is_left ? 0 : strides[0];
    operand.col_stride = is_left ? strides[0] : 0;
    return operand;
  }

  if (trans_batch) {
    if (shape.size() <= 2) {
      throw std::runtime_error("MatMul transBatch requires rank > 2 inputs");
    }
    operand.batch_shape =
        std::vector<int64_t>(shape.begin() + 1, shape.end() - 1);
    operand.batch_strides =
        std::vector<int64_t>(strides.begin() + 1, strides.end() - 1);
    const int64_t leading_dim = shape[0];
    const int64_t trailing_dim = shape.back();
    const int64_t leading_stride = strides[0];
    const int64_t trailing_stride = strides.back();
    operand.rows = trans ? trailing_dim : leading_dim;
    operand.cols = trans ? leading_dim : trailing_dim;
    operand.row_stride = trans ? trailing_stride : leading_stride;
    operand.col_stride = trans ? leading_stride : trailing_stride;
    return operand;
  }

  operand.batch_shape = PrefixShape(shape, 2);
  operand.batch_strides =
      std::vector<int64_t>(strides.begin(), strides.end() - 2);
  const int64_t base_rows = shape[shape.size() - 2];
  const int64_t base_cols = shape[shape.size() - 1];
  const int64_t base_row_stride = strides[shape.size() - 2];
  const int64_t base_col_stride = strides[shape.size() - 1];
  operand.rows = trans ? base_cols : base_rows;
  operand.cols = trans ? base_rows : base_cols;
  operand.row_stride = trans ? base_col_stride : base_row_stride;
  operand.col_stride = trans ? base_row_stride : base_col_stride;
  return operand;
}

MatMulShapeInfo ResolveMatMulShape(const std::vector<int64_t>& a_shape,
                                   const std::vector<int64_t>& b_shape,
                                   bool trans_a, bool trans_b,
                                   bool trans_batch_a, bool trans_batch_b) {
  if (a_shape.empty() || b_shape.empty()) {
    throw std::runtime_error("MatMul input rank must be >= 1");
  }
  if ((trans_batch_a || trans_batch_b) &&
      (a_shape.size() <= 2 || a_shape.size() != b_shape.size())) {
    throw std::runtime_error(
        "MatMul transBatch requires same input rank and rank > 2");
  }

  MatMulShapeInfo info;
  info.lhs_vector = a_shape.size() == 1;
  info.rhs_vector = b_shape.size() == 1;
  if (info.lhs_vector) {
    trans_a = false;
  }
  if (info.rhs_vector) {
    trans_b = false;
  }
  info.lhs = BuildOperand(a_shape, trans_a, trans_batch_a, true);
  info.rhs = BuildOperand(b_shape, trans_b, trans_batch_b, false);
  if (info.lhs.cols != info.rhs.rows) {
    throw std::runtime_error("MatMul K dimension mismatch");
  }

  info.batch_shape =
      BroadcastShapeLocal(info.lhs.batch_shape, info.rhs.batch_shape);
  info.compute_shape = info.batch_shape;
  info.compute_shape.push_back(info.lhs.rows);
  info.compute_shape.push_back(info.rhs.cols);

  info.output_shape = info.batch_shape;
  if (info.lhs_vector && info.rhs_vector) {
    if (!info.output_shape.empty()) {
      throw std::runtime_error("vector-vector MatMul cannot have batch dims");
    }
  } else if (info.lhs_vector) {
    info.output_shape.push_back(info.rhs.cols);
  } else if (info.rhs_vector) {
    info.output_shape.push_back(info.lhs.rows);
  } else {
    info.output_shape.push_back(info.lhs.rows);
    info.output_shape.push_back(info.rhs.cols);
  }
  return info;
}

bool TryMudnnMatMul(void* y_data, const void* a_data, const void* b_data,
                    const std::vector<int64_t>& a_shape,
                    const std::vector<int64_t>& b_shape,
                    const std::vector<int64_t>& y_shape,
                    ONNXTensorElementDataType elem_type, bool trans_a,
                    bool trans_b) {
  std::string key = std::to_string(static_cast<int>(elem_type));
  key += trans_a ? "|ta1" : "|ta0";
  key += trans_b ? "|tb1" : "|tb0";
  AppendShapeKey(key, a_shape);
  AppendShapeKey(key, b_shape);
  AppendShapeKey(key, y_shape);
  static thread_local std::unordered_set<std::string> unsupported_keys;
  if (unsupported_keys.count(key) != 0) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor a_tensor;
  ::musa::dnn::Tensor b_tensor;
  ::musa::dnn::Tensor y_tensor;
  if (!SetMudnnTensor(a_tensor, a_data, a_shape, elem_type) ||
      !SetMudnnTensor(b_tensor, b_data, b_shape, elem_type) ||
      !SetMudnnTensor(y_tensor, y_data, y_shape, elem_type)) {
    return false;
  }

  ::musa::dnn::MatMul matmul;
  if (matmul.SetTranspose(trans_a, trans_b) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  if (matmul.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  if (matmul.SetAlpha(1.0) != ::musa::dnn::Status::SUCCESS) return false;
  if (matmul.SetBeta(0.0) != ::musa::dnn::Status::SUCCESS) return false;
  const bool ok = matmul.Run(*handle, y_tensor, a_tensor, b_tensor) ==
                  ::musa::dnn::Status::SUCCESS;
  if (!ok) unsupported_keys.insert(key);
  return ok;
}

bool TryMublasMatMul(const void* a_data, const void* b_data, void* y_data,
                     const std::vector<int64_t>& a_shape,
                     const std::vector<int64_t>& b_shape,
                     const MatMulShapeInfo& shape_info,
                     ONNXTensorElementDataType elem_type, bool trans_a,
                     bool trans_b, bool trans_batch_a, bool trans_batch_b,
                     float alpha) {
  if (shape_info.lhs_vector || shape_info.rhs_vector || trans_batch_a ||
      trans_batch_b || shape_info.lhs.rows <= 0 || shape_info.rhs.cols <= 0 ||
      shape_info.lhs.cols <= 0) {
    return false;
  }
  const int64_t m64 = shape_info.lhs.rows;
  const int64_t k64 = shape_info.lhs.cols;
  const int64_t n64 = shape_info.rhs.cols;
  if (m64 > INT32_MAX || k64 > INT32_MAX || n64 > INT32_MAX ||
      a_shape.back() > INT32_MAX || b_shape.back() > INT32_MAX) {
    return false;
  }

  mublasHandle_t handle = nullptr;
  OrtStatus* status = EnsureMublasHandle(&handle);
  if (status != nullptr) {
    Ort::GetApi().ReleaseStatus(status);
    return false;
  }

  const int m = static_cast<int>(m64);
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int lda = static_cast<int>(a_shape.back());
  const int ldb = static_cast<int>(b_shape.back());
  const mublasOperation_t op_a = trans_a ? MUBLAS_OP_T : MUBLAS_OP_N;
  const mublasOperation_t op_b = trans_b ? MUBLAS_OP_T : MUBLAS_OP_N;
  const int64_t batch_total = NumElementsLocal(shape_info.batch_shape);

  long long int stride_a = 0;
  long long int stride_b = 0;
  const bool can_use_strided_batched =
      batch_total > 1 && batch_total <= INT32_MAX &&
      TryGetLinearBatchStride(
          shape_info.lhs.batch_shape, shape_info.batch_shape,
          a_shape[a_shape.size() - 2] * a_shape.back(), stride_a) &&
      TryGetLinearBatchStride(
          shape_info.rhs.batch_shape, shape_info.batch_shape,
          b_shape[b_shape.size() - 2] * b_shape.back(), stride_b);
  if (can_use_strided_batched) {
    const long long int stride_y = static_cast<long long int>(m64 * n64);
    mublasStatus gemm_status = MublasGemmStridedBatchedEx(
        handle, op_b, op_a, n, m, k, alpha, b_data, ldb, stride_b, a_data, lda,
        stride_a, 0.0, y_data, n, stride_y, static_cast<int>(batch_total),
        elem_type);
    return gemm_status == MUBLAS_STATUS_SUCCESS;
  }

  if (batch_total == 1) {
    mublasStatus gemm_status =
        MublasGemmEx(handle, op_b, op_a, n, m, k, alpha, b_data, ldb, a_data,
                     lda, 0.0, y_data, n, elem_type);
    return gemm_status == MUBLAS_STATUS_SUCCESS;
  }

  return false;
}

bool TryMudnnBatchMatMul(const void* a_data, const void* b_data, void* y_data,
                         const std::vector<int64_t>& a_shape,
                         const std::vector<int64_t>& b_shape,
                         const MatMulShapeInfo& shape_info,
                         ONNXTensorElementDataType elem_type, bool trans_a,
                         bool trans_b, bool trans_batch_a,
                         bool trans_batch_b, float alpha) {
  if ((elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
       elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
      shape_info.lhs_vector || shape_info.rhs_vector || trans_batch_a ||
      trans_batch_b || alpha != 1.0f || shape_info.lhs.rows <= 0 ||
      shape_info.rhs.cols <= 0 || shape_info.lhs.cols <= 0) {
    return false;
  }

  const int64_t batch_total = NumElementsLocal(shape_info.batch_shape);
  if (batch_total <= 1) {
    return false;
  }

  long long int stride_a = 0;
  long long int stride_b = 0;
  if (!TryGetLinearBatchStride(
          shape_info.lhs.batch_shape, shape_info.batch_shape,
          a_shape[a_shape.size() - 2] * a_shape.back(), stride_a) ||
      !TryGetLinearBatchStride(
          shape_info.rhs.batch_shape, shape_info.batch_shape,
          b_shape[b_shape.size() - 2] * b_shape.back(), stride_b) ||
      stride_a == 0 || stride_b == 0) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  const int64_t m = shape_info.lhs.rows;
  const int64_t n = shape_info.rhs.cols;
  const int64_t k = shape_info.lhs.cols;
  const int64_t lda = a_shape.back();
  const int64_t ldb = b_shape.back();
  const int64_t ldc = n;
  const int64_t stride_c = m * n;

  ::musa::dnn::Tensor a_tensor;
  ::musa::dnn::Tensor b_tensor;
  ::musa::dnn::Tensor y_tensor;
  if (!SetMudnnTensor(a_tensor, a_data, {batch_total, a_shape[a_shape.size() - 2],
                                         a_shape.back()},
                      elem_type) ||
      !SetMudnnTensor(b_tensor, b_data, {batch_total, b_shape[b_shape.size() - 2],
                                         b_shape.back()},
                      elem_type) ||
      !SetMudnnTensor(y_tensor, y_data, {batch_total, m, n}, elem_type)) {
    return false;
  }

  ::musa::dnn::BatchMatMul matmul;
  if (matmul.SetTranspose(trans_a, trans_b) !=
          ::musa::dnn::Status::SUCCESS ||
      matmul.SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR) !=
          ::musa::dnn::Status::SUCCESS ||
      matmul.SetAlpha(1.0) != ::musa::dnn::Status::SUCCESS ||
      matmul.SetBeta(0.0) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }

  return matmul.Run(*handle, y_tensor, a_tensor, b_tensor, batch_total, m, n, k,
                    lda, ldb, ldc, stride_a, stride_b, stride_c) ==
         ::musa::dnn::Status::SUCCESS;
}

MusaBatchedMatMulParams MakeMatMulParams(const MatMulShapeInfo& shape_info,
                                         float alpha) {
  MusaBatchedMatMulParams params{};
  params.output_rank = static_cast<int32_t>(shape_info.compute_shape.size());
  params.batch_rank = static_cast<int32_t>(shape_info.batch_shape.size());
  params.total_elements = NumElementsLocal(shape_info.compute_shape);
  params.m = shape_info.lhs.rows;
  params.n = shape_info.rhs.cols;
  params.k = shape_info.lhs.cols;
  params.alpha = alpha;

  auto output_strides = StridesLocal(shape_info.compute_shape);
  for (size_t dim = 0; dim < shape_info.compute_shape.size(); ++dim) {
    params.output_dims[dim] = shape_info.compute_shape[dim];
    params.output_strides[dim] = output_strides[dim];
  }

  FillBroadcastBatchStrides(shape_info.lhs.batch_shape,
                            shape_info.lhs.batch_strides,
                            shape_info.batch_shape, params.a_batch_strides);
  FillBroadcastBatchStrides(shape_info.rhs.batch_shape,
                            shape_info.rhs.batch_strides,
                            shape_info.batch_shape, params.b_batch_strides);

  params.a_row_stride = shape_info.lhs.row_stride;
  params.a_col_stride = shape_info.lhs.col_stride;
  params.b_row_stride = shape_info.rhs.row_stride;
  params.b_col_stride = shape_info.rhs.col_stride;
  return params;
}

OrtStatus* ComputeMusaMatMulDeviceImpl(
    const void* a_data, const void* b_data, void* y_data,
    ONNXTensorElementDataType elem_type, const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape, const std::vector<int64_t>& y_shape,
    bool trans_a, bool trans_b, bool trans_batch_a, bool trans_batch_b,
    float alpha) {
  if (a_data == nullptr || b_data == nullptr || y_data == nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul device pointers must be non-null.");
  }
  if (!IsMatMulType(elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported MatMul dtype");
  }

  MatMulShapeInfo shape_info = ResolveMatMulShape(
      a_shape, b_shape, trans_a, trans_b, trans_batch_a, trans_batch_b);
  if (!SameShape(y_shape, shape_info.output_shape)) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "MUSA MatMul output shape mismatch.");
  }
  if (shape_info.compute_shape.size() > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "MatMul rank exceeds MUSA device limit");
  }
  if (shape_info.lhs.rows < 0 || shape_info.lhs.cols < 0 ||
      shape_info.rhs.cols < 0) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "MatMul dimensions must be nonnegative");
  }
  if (NumElementsLocal(shape_info.compute_shape) == 0) {
    return nullptr;
  }
  if (shape_info.lhs.cols == 0) {
    musaError_t status = musaMemsetAsync(
        y_data, 0, NumElementsLocal(y_shape) * ElementSize(elem_type), nullptr);
    return LaunchStatus(status);
  }

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      !shape_info.lhs_vector && !shape_info.rhs_vector && !trans_batch_a &&
      !trans_batch_b && alpha == 1.0f && shape_info.lhs.rows == 1 &&
      shape_info.rhs.cols == 1 && shape_info.lhs.cols >= 64 &&
      NumElementsLocal(shape_info.batch_shape) >= 128) {
    MusaBatchedMatMulParams params = MakeMatMulParams(shape_info, alpha);
    musaError_t dot_status = LaunchMusaBatchedDotFloatKernel(
        a_data, b_data, y_data, params, nullptr);
    if (dot_status != musaErrorNotSupported) {
      return LaunchStatus(dot_status);
    }
  }

  if (TryMudnnBatchMatMul(a_data, b_data, y_data, a_shape, b_shape, shape_info,
                          elem_type, trans_a, trans_b, trans_batch_a,
                          trans_batch_b, alpha)) {
    return nullptr;
  }

  const bool prefer_mublas = ResolveTF32EnabledForMublas();
  bool tried_mublas_first = false;
  if (prefer_mublas) {
    tried_mublas_first = true;
    if (TryMublasMatMul(a_data, b_data, y_data, a_shape, b_shape, shape_info,
                        elem_type, trans_a, trans_b, trans_batch_a,
                        trans_batch_b, alpha)) {
      return nullptr;
    }
  }

  const bool can_try_mudnn =
      !shape_info.lhs_vector && !shape_info.rhs_vector && !trans_batch_a &&
      !trans_batch_b && alpha == 1.0f &&
      NumElementsLocal(shape_info.compute_shape) >= 1000000;
  if (can_try_mudnn &&
      TryMudnnMatMul(y_data, a_data, b_data, a_shape, b_shape,
                     shape_info.compute_shape, elem_type, trans_a, trans_b)) {
    return nullptr;
  }

  if (!tried_mublas_first &&
      TryMublasMatMul(a_data, b_data, y_data, a_shape, b_shape, shape_info,
                      elem_type, trans_a, trans_b, trans_batch_a,
                      trans_batch_b, alpha)) {
    return nullptr;
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "unsupported MatMul dtype");
  }
  MusaBatchedMatMulParams params = MakeMatMulParams(shape_info, alpha);
  return LaunchStatus(LaunchMusaBatchedMatMulKernel(
      a_data, b_data, y_data, params, musa_elem_type, nullptr));
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

  auto a_info = a.GetTensorTypeAndShapeInfo();
  auto elem_type = a_info.GetElementType();
  if (b.GetTensorTypeAndShapeInfo().GetElementType() != elem_type) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "MatMul input dtypes must match");
  }
  MatMulShapeInfo shape_info =
      ResolveMatMulShape(a_shape, b_shape, false, false, false, false);
  Ort::UnownedValue y = kernel_context.GetOutput(0, shape_info.output_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "MatMul requires MUSA output");
  }

  return ComputeMusaMatMulDeviceImpl(a.GetTensorRawData(), b.GetTensorRawData(),
                                     y.GetTensorMutableRawData(), elem_type,
                                     a_shape, b_shape, shape_info.output_shape,
                                     false, false, false, false, 1.0f);
}
}  // namespace

OrtStatus* ComputeMusaMatMulOutputShape(const std::vector<int64_t>& a_shape,
                                        const std::vector<int64_t>& b_shape,
                                        bool trans_a, bool trans_b,
                                        bool trans_batch_a, bool trans_batch_b,
                                        std::vector<int64_t>& y_shape) {
  try {
    y_shape = ResolveMatMulShape(a_shape, b_shape, trans_a, trans_b,
                                 trans_batch_a, trans_batch_b)
                  .output_shape;
    return nullptr;
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, ex.what());
  }
}

OrtStatus* ComputeMusaMatMulDevice(
    const void* a_data, const void* b_data, void* y_data,
    ONNXTensorElementDataType elem_type, const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape, const std::vector<int64_t>& y_shape,
    bool trans_a, bool trans_b, bool trans_batch_a, bool trans_batch_b,
    float alpha) {
  return ComputeMusaMatMulDeviceImpl(a_data, b_data, y_data, elem_type, a_shape,
                                     b_shape, y_shape, trans_a, trans_b,
                                     trans_batch_a, trans_batch_b, alpha);
}

OrtStatus* ComputeMusaMatMulDevice(const float* a_data, const float* b_data,
                                   float* y_data,
                                   const std::vector<int64_t>& a_shape,
                                   const std::vector<int64_t>& b_shape,
                                   const std::vector<int64_t>& y_shape) {
  return ComputeMusaMatMulDeviceImpl(
      a_data, b_data, y_data, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, a_shape,
      b_shape, y_shape, false, false, false, false, 1.0f);
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
