// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activation/prelu_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class PRelu : public OpKernelBase<PRelu> {
 public:
  PRelu(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* PRelu::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue slope = ctx.GetInput(1);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto slope_info = slope.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  if (slope_info.GetElementType() != elem_type) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "PRelu input dtypes must match");
  }

  auto input_shape = input_info.GetShape();
  auto slope_shape = slope_info.GetShape();
  std::vector<int64_t> out_shape = BroadcastShape(input_shape, slope_shape);
  if (!CanUseBroadcastKernel(out_shape, input_shape, slope_shape)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "PRelu rank exceeds MUSA device limit");
  }

  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) ||
      !(elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "PRelu unsupported dtype");
  }

  Ort::UnownedValue output = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "PRelu requires MUSA output");
  }

  musaStream_t stream = GetComputeStream(ctx);
  DeviceInputBuffer input_buffer;
  DeviceInputBuffer slope_buffer;
  RETURN_IF_ERROR(input_buffer.Bind(input, stream));
  RETURN_IF_ERROR(slope_buffer.Bind(slope, stream));

  musaError_t status = LaunchMusaPReluKernel(
      input_buffer.data(), slope_buffer.data(),
      output.GetTensorMutableRawData(),
      MakeBroadcastParams(out_shape, input_shape, slope_shape),
      static_cast<int32_t>(elem_type), stream);
  if (status == musaErrorNotSupported) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "PRelu unsupported dtype");
  }
  return LaunchStatus(status);
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    PRelu, kOnnxDomain, 7, 15,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", HfdTensorTypes())), PRelu)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    PRelu, kOnnxDomain, 16, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    PRelu)
