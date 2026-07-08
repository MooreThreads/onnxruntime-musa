// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <string>
#include <vector>

#include "math/cumsum_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

std::vector<const OrtDataType*> CumSumOpset11TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

std::vector<const OrtDataType*> CumSumOpset14TensorTypes() {
  auto types = CumSumOpset11TensorTypes();
  types.push_back(GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  return types;
}

OrtStatus* ReadAxis(Ort::ConstValue axis_tensor, int64_t input_rank,
                    musaStream_t stream, int64_t& axis) {
  auto axis_info = axis_tensor.GetTensorTypeAndShapeInfo();
  auto axis_shape = axis_info.GetShape();
  if (axis_shape.size() > 1 || NumElements(axis_shape) != 1) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CumSum axis tensor must be scalar or 1D with one element");
  }

  auto axis_type = axis_info.GetElementType();
  if (axis_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    axis = static_cast<int64_t>(ReadTyped<int32_t>(axis_tensor, stream)[0]);
  } else if (axis_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    axis = ReadTyped<int64_t>(axis_tensor, stream)[0];
  } else {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT, "CumSum axis tensor must be int32 or int64");
  }

  axis = NormalizeAxis(axis, static_cast<size_t>(input_rank));
  if (axis < 0 || axis >= input_rank) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "CumSum axis out of range");
  }
  return nullptr;
}

bool IsCumSumDeviceType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

class CumSum : public OpKernelBase<CumSum> {
 public:
  CumSum(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    exclusive_ = AttrOrDefault<int64_t>(kernel_info, "exclusive", 0) != 0;
    reverse_ = AttrOrDefault<int64_t>(kernel_info, "reverse", 0) != 0;
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  bool exclusive_ = false;
  bool reverse_ = false;
};

OrtStatus* CumSum::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  Ort::ConstValue axis_input = ctx.GetInput(1);
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto elem_type = input_info.GetElementType();
  auto input_shape = input_info.GetShape();
  if (input_shape.empty()) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Cannot apply CumSum on a scalar");
  }
  for (int64_t dim : input_shape) {
    if (dim < 0) {
      return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "CumSum requires concrete input shape");
    }
  }
  if (!IsCumSumDeviceType(elem_type)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "CumSum unsupported dtype");
  }

  musaStream_t stream = GetComputeStream(ctx);
  int64_t axis = 0;
  RETURN_IF_ERROR(ReadAxis(axis_input, static_cast<int64_t>(input_shape.size()),
                           stream, axis));

  Ort::UnownedValue output = ctx.GetOutput(0, input_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "CumSum requires MUSA output");
  }

  const int64_t output_size = NumElements(input_shape);
  if (output_size == 0) {
    return nullptr;
  }

  DeviceInputBuffer input_buffer;
  RETURN_IF_ERROR(input_buffer.Bind(input, stream));

  int64_t axis_stride = 1;
  for (size_t i = static_cast<size_t>(axis) + 1; i < input_shape.size(); ++i) {
    axis_stride *= input_shape[i];
  }

  return LaunchStatus(LaunchMusaCumSumKernel(
      input_buffer.data(), output.GetTensorMutableRawData(),
      static_cast<int32_t>(elem_type), output_size,
      input_shape[static_cast<size_t>(axis)], axis_stride, exclusive_, reverse_,
      stream));
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    CumSum, kOnnxDomain, 11, 13,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", CumSumOpset11TensorTypes())
         .AddTypeConstraint("T2", IntTensorTypes())),
    CumSum)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    CumSum, kOnnxDomain, 14, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", CumSumOpset14TensorTypes())
         .AddTypeConstraint("T2", IntTensorTypes())),
    CumSum)
