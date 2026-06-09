// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnMinBinary(const void* lhs_data,
                       const std::vector<int64_t>& lhs_shape,
                       const void* rhs_data,
                       const std::vector<int64_t>& rhs_shape, void* output_data,
                       const std::vector<int64_t>& output_shape,
                       ONNXTensorElementDataType elem_type,
                       musaStream_t stream) {
  if (output_shape.size() > kMudnnMaxElementwiseRank ||
      lhs_shape.size() > kMudnnMaxElementwiseRank ||
      rhs_shape.size() > kMudnnMaxElementwiseRank) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, stream);
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
  if (op.SetMode(::musa::dnn::Binary::Mode::MIN) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

bool TryMudnnMin(Ort::KernelContext& ctx,
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

  musaStream_t stream = GetComputeStream(ctx);
  void* output_data = y.GetTensorMutableRawData();
  if (!TryMudnnMinBinary(ctx.GetInput(0).GetTensorRawData(), shapes[0],
                         ctx.GetInput(1).GetTensorRawData(), shapes[1],
                         output_data, output_shape, elem_type, stream)) {
    return false;
  }
  for (size_t input_index = 2; input_index < ctx.GetInputCount();
       ++input_index) {
    if (!TryMudnnMinBinary(y.GetTensorRawData(), output_shape,
                           ctx.GetInput(input_index).GetTensorRawData(),
                           shapes[input_index], output_data, output_shape,
                           elem_type, stream)) {
      return false;
    }
  }
  return true;
}

OrtStatus* ComputeDeviceMinFallback(
    Ort::KernelContext& ctx, const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int64_t>& output_shape,
    ONNXTensorElementDataType elem_type) {
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) || !AllGpuInputs(ctx)) {
    return UnsupportedDeviceElementwiseStatus("Min", elem_type);
  }
  for (const auto& shape : shapes) {
    if (!CanUseBroadcastKernel(output_shape, output_shape, shape)) {
      return UnsupportedDeviceElementwiseStatus("Min", elem_type);
    }
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  musaStream_t stream = GetComputeStream(ctx);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("Min", elem_type);
  }
  if (ctx.GetInputCount() == 1) {
    return CopyRawTensor(ctx.GetInput(0), y,
                         NumElements(output_shape) * ElementSize(elem_type),
                         stream);
  }

  auto launch_binary =
      [&](const void* lhs, const std::vector<int64_t>& lhs_shape,
          const void* rhs, const std::vector<int64_t>& rhs_shape) {
        musaError_t status = LaunchMusaBinaryKernel(
            lhs, rhs, y.GetTensorMutableRawData(),
            MakeBroadcastParams(output_shape, lhs_shape, rhs_shape),
            MusaBinaryOp::Min, musa_elem_type, stream);
        if (status == musaErrorNotSupported) {
          return UnsupportedDeviceElementwiseStatus("Min", elem_type);
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

class Min : public OpKernelBase<Min> {
 public:
  Min(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Min::Compute(Ort::KernelContext& ctx) const {
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
  if (TryMudnnMin(ctx, shapes, out_shape, elem_type)) {
    return nullptr;
  }
  return ComputeDeviceMinFallback(ctx, shapes, out_shape, elem_type);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Min, kOnnxDomain, 13, 19,
                                  (Ort::KernelDefBuilder().AddTypeConstraint(
                                      "T", VariadicMinMaxTensorTypes())),
                                  Min)
