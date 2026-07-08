// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"
#include "tensor/transpose_impl.h"

namespace {
bool TryMudnnTranspose(Ort::KernelContext& ctx, Ort::ConstValue input,
                       Ort::UnownedValue output,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& output_shape,
                       const std::vector<int64_t>& perm,
                       ONNXTensorElementDataType elem_type) {
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo()) || input_shape.size() > 8 ||
      input_shape.size() != perm.size()) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(input_tensor, input.GetTensorRawData(), input_shape,
                      elem_type) ||
      !SetMudnnTensor(output_tensor, output.GetTensorMutableRawData(),
                      output_shape, elem_type)) {
    return false;
  }

  ::musa::dnn::Permute op;
  if (op.ConfigDimStride(output_tensor, input_tensor,
                         static_cast<int>(perm.size()),
                         perm.data()) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, input_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class Transpose : public OpKernelBase<Transpose> {
 public:
  Transpose(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    perm_attr_ = AttrsOrEmpty(kernel_info, "perm");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::vector<int64_t> perm_attr_;
};

OrtStatus* Transpose::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  std::vector<int64_t> perm = perm_attr_;
  if (perm.empty()) {
    for (int64_t i = static_cast<int64_t>(shape0.size()) - 1; i >= 0; --i)
      perm.push_back(i);
  }
  std::vector<int64_t> out_shape;
  for (int64_t p : perm) out_shape.push_back(shape0[static_cast<size_t>(p)]);
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Transpose unsupported dtype");
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (NumElements(out_shape) == 0) {
    return nullptr;
  }
  if (TryMudnnTranspose(ctx, input0, y, shape0, out_shape, perm, elem_type)) {
    return nullptr;
  }
  if (shape0.size() <= kMusaMaxBroadcastRank &&
      IsGpuMemory(input0.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    auto in_strides = Strides(shape0);
    MusaTransposeParams params{};
    params.rank = static_cast<int32_t>(shape0.size());
    params.total_elements = NumElements(out_shape);
    for (size_t dim = 0; dim < shape0.size(); ++dim) {
      params.input_strides[dim] = in_strides[dim];
      params.output_dims[dim] = out_shape[dim];
      params.perm[dim] = static_cast<int32_t>(perm[dim]);
    }
    return LaunchStatus(LaunchMusaTransposeKernel(
        input0.GetTensorRawData(), y.GetTensorMutableRawData(),
        static_cast<int32_t>(elem_size), params, GetComputeStream(ctx)));
  }
  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Transpose requires MUSA input/output and supported rank");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", AllFixedSizeTensorTypes())),
    Transpose)
