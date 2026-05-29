#include "kernels/matmul.h"

#include <vector>

#include "kernel_utils.h"

MatMulKernel::MatMulKernel(const OrtKernelInfo* info, PrivateTag)
    : OrtKernelImpl{}, info_(info) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus* MatMulKernel::CreateKernelImpl(const OrtKernelInfo* info, void*,
                                          OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  kernel = new MatMulKernel(info, PrivateTag{});
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL MatMulKernel::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  static_cast<void>(static_cast<MatMulKernel*>(this_ptr)->info_);
  Ort::KernelContext kernel_context(context);

  const float* lhs_data = nullptr;
  const float* rhs_data = nullptr;
  std::vector<int64_t> lhs_shape;
  std::vector<int64_t> rhs_shape;
  size_t lhs_count = 0;
  size_t rhs_count = 0;
  RETURN_IF_ERROR(
      GetInputDataAndShape(kernel_context, 0, lhs_data, lhs_shape, lhs_count));
  RETURN_IF_ERROR(
      GetInputDataAndShape(kernel_context, 1, rhs_data, rhs_shape, rhs_count));
  RETURN_IF(lhs_shape.size() != 2 || rhs_shape.size() != 2, Ort::GetApi(),
            "MatMulKernel currently supports 2-D tensors only.");
  RETURN_IF(lhs_shape[1] != rhs_shape[0], Ort::GetApi(),
            "MatMul input dimensions are incompatible.");

  const int64_t m = lhs_shape[0];
  const int64_t k = lhs_shape[1];
  const int64_t n = rhs_shape[1];
  std::vector<int64_t> output_shape{m, n};
  Ort::UnownedValue output = kernel_context.GetOutput(0, output_shape);
  float* output_data = output.GetTensorMutableData<float>();

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int64_t inner = 0; inner < k; ++inner) {
        acc += lhs_data[row * k + inner] * rhs_data[inner * n + col];
      }
      output_data[row * n + col] = acc;
    }
  }

  static_cast<void>(lhs_count);
  static_cast<void>(rhs_count);
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

void ORT_API_CALL MatMulKernel::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<MatMulKernel*>(this_ptr);
}
