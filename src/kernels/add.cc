#include "kernels/add.h"

#include <vector>

#include "kernel_utils.h"

AddKernel::AddKernel(const OrtKernelInfo* info, PrivateTag)
    : OrtKernelImpl{}, info_(info) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

OrtStatus* AddKernel::CreateKernelImpl(const OrtKernelInfo* info, void*,
                                       OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  kernel = new AddKernel(info, PrivateTag{});
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL AddKernel::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  static_cast<void>(static_cast<AddKernel*>(this_ptr)->info_);
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

  std::vector<int64_t> output_shape;
  RETURN_IF(!BroadcastShapes(lhs_shape, rhs_shape, output_shape), Ort::GetApi(),
            "Add inputs are not broadcastable.");
  Ort::UnownedValue output = kernel_context.GetOutput(0, output_shape);
  float* output_data = output.GetTensorMutableData<float>();

  const int64_t output_count = ShapeSize(output_shape);
  const std::vector<int64_t> output_strides = ContiguousStrides(output_shape);
  const std::vector<int64_t> lhs_strides = ContiguousStrides(lhs_shape);
  const std::vector<int64_t> rhs_strides = ContiguousStrides(rhs_shape);
  const size_t output_rank = output_shape.size();

  for (int64_t linear = 0; linear < output_count; ++linear) {
    int64_t remaining = linear;
    int64_t lhs_offset = 0;
    int64_t rhs_offset = 0;

    for (size_t axis = 0; axis < output_rank; ++axis) {
      const int64_t coordinate =
          output_strides[axis] == 0 ? 0 : remaining / output_strides[axis];
      remaining =
          output_strides[axis] == 0 ? 0 : remaining % output_strides[axis];

      if (axis >= output_rank - lhs_shape.size()) {
        const size_t lhs_axis = axis - (output_rank - lhs_shape.size());
        if (lhs_shape[lhs_axis] != 1) {
          lhs_offset += coordinate * lhs_strides[lhs_axis];
        }
      }
      if (axis >= output_rank - rhs_shape.size()) {
        const size_t rhs_axis = axis - (output_rank - rhs_shape.size());
        if (rhs_shape[rhs_axis] != 1) {
          rhs_offset += coordinate * rhs_strides[rhs_axis];
        }
      }
    }

    output_data[linear] = lhs_data[lhs_offset] + rhs_data[rhs_offset];
  }

  static_cast<void>(lhs_count);
  static_cast<void>(rhs_count);
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

void ORT_API_CALL AddKernel::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<AddKernel*>(this_ptr);
}
