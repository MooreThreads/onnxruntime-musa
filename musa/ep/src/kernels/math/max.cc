// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnMaxBinary(const void* lhs_data,
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

  musaStream_t stream = GetComputeStream(ctx);
  void* output_data = y.GetTensorMutableRawData();
  if (!TryMudnnMaxBinary(ctx.GetInput(0).GetTensorRawData(), shapes[0],
                         ctx.GetInput(1).GetTensorRawData(), shapes[1],
                         output_data, output_shape, elem_type, stream)) {
    return false;
  }
  for (size_t input_index = 2; input_index < ctx.GetInputCount();
       ++input_index) {
    if (!TryMudnnMaxBinary(y.GetTensorRawData(), output_shape,
                           ctx.GetInput(input_index).GetTensorRawData(),
                           shapes[input_index], output_data, output_shape,
                           elem_type, stream)) {
      return false;
    }
  }
  return true;
}

template <typename T>
OrtStatus* ComputeMaxCpuMetadataTyped(
    Ort::KernelContext& ctx, const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int64_t>& output_shape, musaStream_t stream) {
  const int64_t total = NumElements(output_shape);
  if (total > kMusaMaxBroadcastRank) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "Max CPU metadata path only supports small shape tensors");
  }

  std::vector<std::vector<T>> inputs;
  inputs.reserve(ctx.GetInputCount());
  std::vector<std::vector<int64_t>> strides;
  strides.reserve(ctx.GetInputCount());
  for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
    inputs.push_back(ReadTyped<T>(ctx.GetInput(i), stream));
    strides.push_back(Strides(shapes[i]));
  }

  std::vector<T> output(static_cast<size_t>(total));
  for (int64_t i = 0; i < total; ++i) {
    auto coord = Coordinates(i, output_shape);
    T value = inputs[0][static_cast<size_t>(
        BroadcastOffset(coord, shapes[0], strides[0]))];
    for (size_t input_index = 1; input_index < inputs.size(); ++input_index) {
      T next = inputs[input_index][static_cast<size_t>(
          BroadcastOffset(coord, shapes[input_index], strides[input_index]))];
      value = std::max(value, next);
    }
    output[static_cast<size_t>(i)] = value;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  return WriteTyped<T>(y, output, stream);
}

OrtStatus* ComputeMaxCpuMetadata(
    Ort::KernelContext& ctx, const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int64_t>& output_shape,
    ONNXTensorElementDataType elem_type) {
  musaStream_t stream = GetComputeStream(ctx);
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return ComputeMaxCpuMetadataTyped<int64_t>(ctx, shapes, output_shape,
                                               stream);
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return ComputeMaxCpuMetadataTyped<int32_t>(ctx, shapes, output_shape,
                                               stream);
  }
  return Ort::GetApi().CreateStatus(
      ORT_NOT_IMPLEMENTED,
      "Max CPU metadata path only supports int32/int64 tensors");
}

OrtStatus* ComputeDeviceMaxFallback(
    Ort::KernelContext& ctx, const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int64_t>& output_shape,
    ONNXTensorElementDataType elem_type) {
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type) || !AllGpuInputs(ctx)) {
    if (!AllGpuInputs(ctx)) {
      return ComputeMaxCpuMetadata(ctx, shapes, output_shape, elem_type);
    }
    return UnsupportedDeviceElementwiseStatus("Max", elem_type);
  }
  for (const auto& shape : shapes) {
    if (!CanUseBroadcastKernel(output_shape, output_shape, shape)) {
      return UnsupportedDeviceElementwiseStatus("Max", elem_type);
    }
  }

  Ort::UnownedValue y = ctx.GetOutput(0, output_shape);
  musaStream_t stream = GetComputeStream(ctx);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("Max", elem_type);
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
            MusaBinaryOp::Max, musa_elem_type, stream);
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
  if (NumElements(out_shape) == 0) {
    ctx.GetOutput(0, out_shape);
    return nullptr;
  }
  if (TryMudnnMax(ctx, shapes, out_shape, elem_type)) {
    return nullptr;
  }
  return ComputeDeviceMaxFallback(ctx, shapes, out_shape, elem_type);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Max, kOnnxDomain, 13, 19,
                                  (Ort::KernelDefBuilder().AddTypeConstraint(
                                      "T", VariadicMinMaxTensorTypes())),
                                  Max)
