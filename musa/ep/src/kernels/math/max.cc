// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnMaxBinary(const void* lhs_data,
                       const std::vector<int64_t>& lhs_shape,
                       const void* rhs_data,
                       const std::vector<int64_t>& rhs_shape,
                       void* output_data,
                       const std::vector<int64_t>& output_shape,
                       ONNXTensorElementDataType elem_type) {
  if (output_shape.size() > kMudnnMaxElementwiseRank ||
      lhs_shape.size() > kMudnnMaxElementwiseRank ||
      rhs_shape.size() > kMudnnMaxElementwiseRank) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle);
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor lhs_tensor;
  ::musa::dnn::Tensor rhs_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(lhs_tensor, lhs_data, lhs_shape, elem_type) ||
      !SetMudnnTensor(rhs_tensor, rhs_data, rhs_shape, elem_type) ||
      !SetMudnnTensor(output_tensor, output_data, output_shape, elem_type)) {
    return false;
  }

  ::musa::dnn::Binary op;
  if (op.SetMode(::musa::dnn::Binary::Mode::MAX) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

bool TryMudnnMax(Ort::KernelContext& ctx,
                 const std::vector<std::vector<int64_t>>& shapes,
                 const std::vector<int64_t>& output_shape,
                 ONNXTensorElementDataType elem_type) {
  if (ctx.GetInputCount() < 2 || !AllGpuInputs(ctx)) {
    return false;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return false;
  }

  void* output_data = y.GetTensorMutableRawData();
  if (!TryMudnnMaxBinary(ctx.GetInput(0).GetTensorRawData(), shapes[0],
                         ctx.GetInput(1).GetTensorRawData(), shapes[1],
                         output_data, output_shape, elem_type)) {
    return false;
  }
  for (size_t input_index = 2; input_index < ctx.GetInputCount();
       ++input_index) {
    if (!TryMudnnMaxBinary(y.GetTensorRawData(), output_shape,
                           ctx.GetInput(input_index).GetTensorRawData(),
                           shapes[input_index], output_data, output_shape,
                           elem_type)) {
      return false;
    }
  }
  return true;
}

OrtStatus* TryScalarInt32Max(
    Ort::KernelContext& ctx, const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int64_t>& output_shape, ONNXTensorElementDataType elem_type,
    bool& handled) {
  handled = false;
  if (ctx.GetInputCount() != 2 ||
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
      !AllGpuInputs(ctx)) {
    return nullptr;
  }
  const bool lhs_scalar = shapes[0].empty();
  const bool rhs_scalar = shapes[1].empty();
  if (lhs_scalar == rhs_scalar) {
    return nullptr;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("Max", elem_type);
  }

  handled = true;
  return LaunchStatus(LaunchMusaBinaryScalarInt32Kernel(
      ctx.GetInput(0).GetTensorData<int32_t>(),
      ctx.GetInput(1).GetTensorData<int32_t>(), y.GetTensorMutableData<int32_t>(),
      NumElements(output_shape), lhs_scalar, MusaBinaryOp::Max, nullptr));
}

OrtStatus* ComputeDeviceMaxFallback(
    Ort::KernelContext& ctx, const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int64_t>& output_shape,
    ONNXTensorElementDataType elem_type) {
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) || !AllGpuInputs(ctx)) {
    return UnsupportedDeviceElementwiseStatus("Max", elem_type);
  }
  for (const auto& shape : shapes) {
    if (!CanUseBroadcastKernel(output_shape, output_shape, shape)) {
      return UnsupportedDeviceElementwiseStatus("Max", elem_type);
    }
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("Max", elem_type);
  }
  if (ctx.GetInputCount() == 1) {
    return CopyRawTensor(ctx.GetInput(0), y,
                         NumElements(output_shape) * ElementSize(elem_type));
  }

  auto launch_binary = [&](const void* lhs, const std::vector<int64_t>& lhs_shape,
                           const void* rhs,
                           const std::vector<int64_t>& rhs_shape) {
    musaError_t status = LaunchMusaBinaryKernel(
        lhs, rhs, y.GetTensorMutableRawData(),
        MakeBroadcastParams(output_shape, lhs_shape, rhs_shape),
        MusaBinaryOp::Max, musa_elem_type, nullptr);
    if (status == musaErrorNotSupported) {
      return UnsupportedDeviceElementwiseStatus("Max", elem_type);
    }
    return LaunchStatus(status);
  };

  RETURN_IF_ERROR(launch_binary(ctx.GetInput(0).GetTensorRawData(), shapes[0],
                                ctx.GetInput(1).GetTensorRawData(), shapes[1]));
  for (size_t input_index = 2; input_index < ctx.GetInputCount();
       ++input_index) {
    RETURN_IF_ERROR(launch_binary(y.GetTensorRawData(), output_shape,
                                  ctx.GetInput(input_index).GetTensorRawData(),
                                  shapes[input_index]));
  }
  return nullptr;
}

class Max : public OpKernelBase<Max> {
 public:
  Max(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Max::Compute(Ort::KernelContext& ctx) const {
  auto elem_type = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetElementType();
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(ctx.GetInputCount());
  std::vector<int64_t> out_shape =
      ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
  for (size_t input_index = 0; input_index < ctx.GetInputCount();
       ++input_index) {
    shapes.push_back(
        ctx.GetInput(input_index).GetTensorTypeAndShapeInfo().GetShape());
    out_shape = BroadcastShape(out_shape, shapes.back());
  }
  bool handled = false;
  OrtStatus* scalar_status = TryScalarInt32Max(ctx, shapes, out_shape, elem_type,
                                               handled);
  if (handled) {
    return scalar_status;
  }
  if (TryMudnnMax(ctx, shapes, out_shape, elem_type)) {
    return nullptr;
  }
  return ComputeDeviceMaxFallback(ctx, shapes, out_shape, elem_type);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Max, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", VariadicMinMaxTensorTypes())),
    Max)
