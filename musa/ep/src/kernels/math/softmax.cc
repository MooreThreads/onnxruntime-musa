// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"
#include "math/softmax_impl.h"

namespace {
bool IsSoftmaxDeviceType(MusaElementType elem_type) {
  return elem_type == MusaElementType::Float ||
         elem_type == MusaElementType::Double ||
         elem_type == MusaElementType::Float16 ||
         elem_type == MusaElementType::BFloat16;
}

bool TryMudnnSoftmax(Ort::KernelContext& ctx,
                     const std::vector<int64_t>& shape,
                     int64_t axis,
                     ONNXTensorElementDataType elem_type,
                     Ort::UnownedValue y) {
  Ort::ConstValue input = ctx.GetInput(0);
  if (!IsGpuMemory(input.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor input_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(input_tensor, input.GetTensorRawData(), shape,
                      elem_type) ||
      !SetMudnnTensor(output_tensor, y.GetTensorMutableRawData(), shape,
                      elem_type)) {
    return false;
  }

  ::musa::dnn::Softmax op;
  if (op.SetMode(::musa::dnn::Softmax::Mode::SOFTMAX) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE) !=
          ::musa::dnn::Status::SUCCESS ||
      op.SetDim(static_cast<int>(axis)) != ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, input_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class Softmax : public OpKernelBase<Softmax> {
 public:
  Softmax(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", -1);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Softmax::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !IsSoftmaxDeviceType(musa_elem_type)) {
    return UnsupportedDeviceElementwiseStatus("Softmax", elem_type);
  }
  std::vector<int64_t> shape0 = info.GetShape();
  int64_t axis = 0;
  int64_t outer = 1;
  int64_t dim = 1;
  int64_t inner = 1;
  if (!shape0.empty()) {
    axis = NormalizeAxis(axis_, shape0.size());
    if (axis < 0 || axis >= static_cast<int64_t>(shape0.size())) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Softmax axis out of range");
    }
    dim = shape0[static_cast<size_t>(axis)];
    for (int64_t i = 0; i < axis; ++i) {
      outer *= shape0[static_cast<size_t>(i)];
    }
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape0.size(); ++i) {
      inner *= shape0[i];
    }
  }
  Ort::UnownedValue y = ctx.GetOutput(0, shape0);
  if (TryMudnnSoftmax(ctx, shape0, axis, elem_type, y)) {
    return nullptr;
  }

  if (!IsGpuMemory(input0.GetTensorMemoryInfo()) ||
      !IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("Softmax", elem_type);
  }
  return LaunchStatus(LaunchMusaSoftmaxKernel(
      input0.GetTensorRawData(), y.GetTensorMutableRawData(), outer, dim,
      inner, musa_elem_type, nullptr));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Softmax)
