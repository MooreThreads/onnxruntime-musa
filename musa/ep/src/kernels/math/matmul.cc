#include "matmul.h"

#include <mudnncxx/mudnn.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<int64_t> BroadcastShapeLocal(const std::vector<int64_t>& a,
                                         const std::vector<int64_t>& b) {
  size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      throw std::runtime_error("MatMul batch broadcast shape mismatch");
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

bool ResolveLinearBatchStride(const std::vector<int64_t>& input_batch,
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

OrtStatus* MudnnStatus(::musa::dnn::Status status, const char* message) {
  if (status == ::musa::dnn::Status::SUCCESS) {
    return nullptr;
  }
  std::string error = std::string(message) +
                      ", status=" + std::to_string(static_cast<int>(status));
  return Ort::GetApi().CreateStatus(ORT_EP_FAIL, error.c_str());
}

OrtStatus* SetMudnnBatchedMatMulTensor(::musa::dnn::Tensor& tensor,
                                       const void* data,
                                       const std::vector<int64_t>& shape,
                                       ONNXTensorElementDataType elem_type,
                                       int64_t batch, int64_t batch_stride) {
  if (!SetMudnnTensor(tensor, data, shape, elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "failed to setup muDNN MatMul tensor");
  }

  const int64_t rows = shape[shape.size() - 2];
  const int64_t cols = shape[shape.size() - 1];
  const int64_t dims[3] = {batch, rows, cols};
  const int64_t strides[3] = {batch_stride, cols, 1};
  return MudnnStatus(tensor.SetNdInfo(3, dims, strides),
                     "failed to set muDNN BatchMatMul tensor shape");
}

OrtStatus* RunMudnnMatMul(void* y_data, const void* a_data, const void* b_data,
                          const std::vector<int64_t>& a_shape,
                          const std::vector<int64_t>& b_shape,
                          const std::vector<int64_t>& y_shape,
                          ONNXTensorElementDataType elem_type, bool trans_a,
                          bool trans_b, float alpha, musaStream_t stream) {
  ::musa::dnn::Handle* handle = nullptr;
  RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));

  ::musa::dnn::Tensor a_tensor;
  ::musa::dnn::Tensor b_tensor;
  ::musa::dnn::Tensor y_tensor;
  if (!SetMudnnTensor(a_tensor, a_data, a_shape, elem_type) ||
      !SetMudnnTensor(b_tensor, b_data, b_shape, elem_type) ||
      !SetMudnnTensor(y_tensor, y_data, y_shape, elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                      "failed to setup muDNN MatMul tensors");
  }

  ::musa::dnn::MatMul matmul;
  RETURN_IF_ERROR(MudnnStatus(matmul.SetTranspose(trans_a, trans_b),
                              "muDNN MatMul SetTranspose failed"));
  RETURN_IF_ERROR(MudnnStatus(
      matmul.SetComputeMode(::musa::dnn::MatMul::ComputeMode::TENSOR),
      "muDNN MatMul SetComputeMode failed"));
  RETURN_IF_ERROR(MudnnStatus(matmul.SetAlpha(static_cast<double>(alpha)),
                              "muDNN MatMul SetAlpha failed"));
  RETURN_IF_ERROR(
      MudnnStatus(matmul.SetBeta(0.0), "muDNN MatMul SetBeta failed"));
  return MudnnStatus(matmul.Run(*handle, y_tensor, a_tensor, b_tensor),
                     "muDNN MatMul execution failed");
}

OrtStatus* RunMudnnBatchMatMul(
    void* y_data, const void* a_data, const void* b_data,
    const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
    const MatMulShapeInfo& shape_info, ONNXTensorElementDataType elem_type,
    bool trans_a, bool trans_b, float alpha, musaStream_t stream) {
  const int64_t batch_total = NumElementsLocal(shape_info.batch_shape);
  long long int stride_a = 0;
  long long int stride_b = 0;
  if (!ResolveLinearBatchStride(
          shape_info.lhs.batch_shape, shape_info.batch_shape,
          a_shape[a_shape.size() - 2] * a_shape.back(), stride_a) ||
      !ResolveLinearBatchStride(
          shape_info.rhs.batch_shape, shape_info.batch_shape,
          b_shape[b_shape.size() - 2] * b_shape.back(), stride_b)) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "muDNN BatchMatMul only supports equal or single-batch broadcast");
  }

  ::musa::dnn::Handle* handle = nullptr;
  RETURN_IF_ERROR(EnsureMudnnHandle(&handle, stream));

  ::musa::dnn::Tensor a_tensor;
  ::musa::dnn::Tensor b_tensor;
  ::musa::dnn::Tensor y_tensor;
  const int64_t a_batch = stride_a == 0 ? 1 : batch_total;
  const int64_t b_batch = stride_b == 0 ? 1 : batch_total;
  const std::vector<int64_t> y_shape_3d = {batch_total, shape_info.lhs.rows,
                                           shape_info.rhs.cols};
  RETURN_IF_ERROR(SetMudnnBatchedMatMulTensor(a_tensor, a_data, a_shape,
                                              elem_type, a_batch, stride_a));
  RETURN_IF_ERROR(SetMudnnBatchedMatMulTensor(b_tensor, b_data, b_shape,
                                              elem_type, b_batch, stride_b));
  RETURN_IF_ERROR(SetMudnnBatchedMatMulTensor(
      y_tensor, y_data, y_shape_3d, elem_type, batch_total,
      shape_info.lhs.rows * shape_info.rhs.cols));

  ::musa::dnn::BatchMatMul batch_op;
  RETURN_IF_ERROR(MudnnStatus(batch_op.SetTranspose(trans_a, trans_b),
                              "muDNN BatchMatMul SetTranspose failed"));
  RETURN_IF_ERROR(MudnnStatus(
      batch_op.SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR),
      "muDNN BatchMatMul SetComputeMode failed"));
  RETURN_IF_ERROR(MudnnStatus(batch_op.SetAlpha(static_cast<double>(alpha)),
                              "muDNN BatchMatMul SetAlpha failed"));
  RETURN_IF_ERROR(
      MudnnStatus(batch_op.SetBeta(0.0), "muDNN BatchMatMul SetBeta failed"));

  ::musa::dnn::MemoryMaintainer maintainer =
      [stream](size_t bytes) -> ::musa::dnn::MemoryHandler {
    void* ptr = nullptr;
    if (bytes != 0 && musaMalloc(&ptr, bytes) != musaSuccess) {
      ptr = nullptr;
    }
    return ::musa::dnn::MemoryHandler(ptr, [stream](void* p) {
      FreeDeviceMemoryOnStream(p, stream);
    });
  };

  RETURN_IF_ERROR(MudnnStatus(
      batch_op.Run(*handle, y_tensor, a_tensor, b_tensor, maintainer),
      "muDNN BatchMatMul execution failed"));
  return nullptr;
}

OrtStatus* ComputeMusaMatMulDeviceImpl(
    const void* a_data, const void* b_data, void* y_data,
    ONNXTensorElementDataType elem_type, const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape, const std::vector<int64_t>& y_shape,
    bool trans_a, bool trans_b, bool trans_batch_a, bool trans_batch_b,
    float alpha, musaStream_t stream) {
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
  if (y_data == nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul output pointer must be non-null.");
  }
  if (shape_info.lhs.cols == 0) {
    musaError_t status = musaMemsetAsync(
        y_data, 0, NumElementsLocal(y_shape) * ElementSize(elem_type), stream);
    return LaunchStatus(status);
  }
  if (a_data == nullptr || b_data == nullptr) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "MUSA MatMul input pointers must be non-null.");
  }
  if (shape_info.lhs_vector || shape_info.rhs_vector) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "muDNN MatMul port does not support ONNX vector MatMul inputs");
  }
  if (trans_batch_a || trans_batch_b) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "muDNN MatMul port does not support transBatch attributes");
  }

  if (shape_info.batch_shape.empty()) {
    return RunMudnnMatMul(y_data, a_data, b_data, a_shape, b_shape,
                          shape_info.compute_shape, elem_type, trans_a, trans_b,
                          alpha, stream);
  }

  return RunMudnnBatchMatMul(y_data, a_data, b_data, a_shape, b_shape,
                             shape_info, elem_type, trans_a, trans_b, alpha,
                             stream);
}

OrtStatus* ComputeMudnnMatMul(Ort::KernelContext& kernel_context,
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

  return ComputeMusaMatMulDeviceImpl(
      a.GetTensorRawData(), b.GetTensorRawData(), y.GetTensorMutableRawData(),
      elem_type, a_shape, b_shape, shape_info.output_shape, false, false, false,
      false, 1.0f, GetComputeStream(kernel_context));
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
    float alpha, musaStream_t stream) {
  return ComputeMusaMatMulDeviceImpl(
      a_data, b_data, y_data, elem_type, a_shape, b_shape, y_shape, trans_a,
      trans_b, trans_batch_a, trans_batch_b, alpha, stream);
}

OrtStatus* ComputeMusaMatMulDevice(const float* a_data, const float* b_data,
                                   float* y_data,
                                   const std::vector<int64_t>& a_shape,
                                   const std::vector<int64_t>& b_shape,
                                   const std::vector<int64_t>& y_shape,
                                   musaStream_t stream) {
  return ComputeMusaMatMulDeviceImpl(
      a_data, b_data, y_data, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, a_shape,
      b_shape, y_shape, false, false, false, false, 1.0f, stream);
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

  return ComputeMudnnMatMul(kernel_context, a, b, a_shape, b_shape);
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL MatMul::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete reinterpret_cast<MatMul*>(this_ptr);
}
