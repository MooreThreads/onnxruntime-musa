// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"
#include "math/variadic_elementwise_ops_impl.h"

namespace {
// Combines ORT CUDA EP's variadic_elementwise_ops Sum control flow with
// MUSA's preferred muDNN backend: one input is a device copy, same-shape and
// broadcasted inputs try muDNN Binary ADD first, then fall back to the existing
// MUSA binary device kernel when muDNN is not applicable.
constexpr size_t kNoInputIndex = std::numeric_limits<size_t>::max();
constexpr size_t kMudnnMaxSumRank = 5;

bool AllSameShape(const std::vector<std::vector<int64_t>>& input_shapes) {
  if (input_shapes.empty()) {
    return false;
  }
  const auto& shape0 = input_shapes[0];
  return std::all_of(
      input_shapes.begin() + 1, input_shapes.end(),
      [&shape0](const std::vector<int64_t>& shape) { return shape == shape0; });
}

bool CanUseMudnnSumShapes(const std::vector<std::vector<int64_t>>& input_shapes,
                          const std::vector<int64_t>& output_shape) {
  if (input_shapes.empty() || output_shape.size() > kMudnnMaxSumRank) {
    return false;
  }
  return std::all_of(input_shapes.begin(), input_shapes.end(),
                     [](const std::vector<int64_t>& shape) {
                       return shape.size() <= kMudnnMaxSumRank;
                     });
}

bool CanUseBroadcastSumKernel(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<int64_t>& output_shape) {
  if (output_shape.size() > kMusaMaxBroadcastRank) {
    return false;
  }
  return std::all_of(input_shapes.begin(), input_shapes.end(),
                     [](const std::vector<int64_t>& shape) {
                       return shape.size() <= kMusaMaxBroadcastRank;
                     });
}

size_t FindSameShapeInput(const std::vector<std::vector<int64_t>>& input_shapes,
                          const std::vector<int64_t>& output_shape) {
  for (size_t input_index = 0; input_index < input_shapes.size();
       ++input_index) {
    if (input_shapes[input_index] == output_shape) {
      return input_index;
    }
  }
  return kNoInputIndex;
}

bool TryMudnnAdd(const void* lhs_data, const std::vector<int64_t>& lhs_shape,
                 const void* rhs_data, const std::vector<int64_t>& rhs_shape,
                 void* output_data, const std::vector<int64_t>& output_shape,
                 ONNXTensorElementDataType elem_type) {
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

  ::musa::dnn::Binary add_op;
  if (add_op.SetMode(::musa::dnn::Binary::Mode::ADD) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return add_op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

bool TryMudnnSum(Ort::KernelContext& ctx,
                 const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<int64_t>& output_shape,
                 Ort::UnownedValue y,
                 ONNXTensorElementDataType elem_type) {
  const size_t input_count = ctx.GetInputCount();
  if (input_count < 2 || !CanUseMudnnSumShapes(input_shapes, output_shape)) {
    return false;
  }

  void* output_data = y.GetTensorMutableRawData();
  std::vector<bool> consumed(input_count, false);

  if (AllSameShape(input_shapes)) {
    if (!TryMudnnAdd(ctx.GetInput(0).GetTensorRawData(), input_shapes[0],
                     ctx.GetInput(1).GetTensorRawData(), input_shapes[1],
                     output_data, output_shape, elem_type)) {
      return false;
    }
    consumed[0] = true;
    consumed[1] = true;
  } else {
    size_t same_shape_input = FindSameShapeInput(input_shapes, output_shape);
    if (same_shape_input != kNoInputIndex) {
      const size_t other_input = same_shape_input == 0 ? 1 : 0;
      if (!TryMudnnAdd(ctx.GetInput(same_shape_input).GetTensorRawData(),
                       input_shapes[same_shape_input],
                       ctx.GetInput(other_input).GetTensorRawData(),
                       input_shapes[other_input], output_data, output_shape,
                       elem_type)) {
        return false;
      }
      consumed[same_shape_input] = true;
      consumed[other_input] = true;
    } else {
      if (!TryMudnnAdd(ctx.GetInput(0).GetTensorRawData(), input_shapes[0],
                       ctx.GetInput(1).GetTensorRawData(), input_shapes[1],
                       output_data, output_shape, elem_type)) {
        return false;
      }
      consumed[0] = true;
      consumed[1] = true;
    }
  }

  for (size_t input_index = 0; input_index < input_count; ++input_index) {
    if (consumed[input_index]) {
      continue;
    }
    if (!TryMudnnAdd(y.GetTensorRawData(), output_shape,
                     ctx.GetInput(input_index).GetTensorRawData(),
                     input_shapes[input_index], output_data, output_shape,
                     elem_type)) {
      return false;
    }
  }

  return true;
}

OrtStatus* LaunchBroadcastAdd(const void* lhs_data,
                              const std::vector<int64_t>& lhs_shape,
                              const void* rhs_data,
                              const std::vector<int64_t>& rhs_shape,
                              void* output_data,
                              const std::vector<int64_t>& output_shape,
                              ONNXTensorElementDataType elem_type) {
  MusaElementType musa_elem_type;
  if (!ToMusaElementType(elem_type, musa_elem_type)) {
    return UnsupportedDeviceElementwiseStatus("Sum", elem_type);
  }
  MusaBroadcastParams params =
      MakeBroadcastParams(output_shape, lhs_shape, rhs_shape);
  musaError_t status = LaunchMusaVariadicSumKernel(
      lhs_data, rhs_data, output_data, params, musa_elem_type, nullptr);
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus("Sum", elem_type);
  }
  return LaunchStatus(status);
}

OrtStatus* ComputeDeviceSumFallback(
    Ort::KernelContext& ctx,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<int64_t>& output_shape, Ort::UnownedValue y,
    bool& handled, ONNXTensorElementDataType elem_type) {
  handled = false;
  if (!CanUseBroadcastSumKernel(input_shapes, output_shape)) {
    return nullptr;
  }

  handled = true;
  void* output_data = y.GetTensorMutableRawData();
  RETURN_IF_ERROR(LaunchBroadcastAdd(
      ctx.GetInput(0).GetTensorRawData(), input_shapes[0],
      ctx.GetInput(1).GetTensorRawData(), input_shapes[1], output_data,
      output_shape, elem_type));
  for (size_t input_index = 2; input_index < ctx.GetInputCount();
       ++input_index) {
    RETURN_IF_ERROR(LaunchBroadcastAdd(
        y.GetTensorRawData(), output_shape,
        ctx.GetInput(input_index).GetTensorRawData(),
        input_shapes[input_index], output_data, output_shape, elem_type));
  }
  return nullptr;
}

OrtStatus* ComputeDeviceSum(
    Ort::KernelContext& ctx,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<int64_t>& output_shape, Ort::UnownedValue y,
    bool& handled, ONNXTensorElementDataType elem_type) {
  handled = false;
  if (ctx.GetInputCount() == 1) {
    handled = true;
    Ort::ConstValue input0 = ctx.GetInput(0);
    return CopyRawTensor(input0, y, input0.GetTensorSizeInBytes());
  }

  if (TryMudnnSum(ctx, input_shapes, output_shape, y, elem_type)) {
    handled = true;
    return nullptr;
  }

  return ComputeDeviceSumFallback(ctx, input_shapes, output_shape, y, handled,
                                  elem_type);
}

class Sum : public OpKernelBase<Sum> {
 public:
  Sum(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Sum::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();

  std::vector<int64_t> shape0 = info.GetShape();
  std::vector<std::vector<int64_t>> input_shapes;
  input_shapes.reserve(ctx.GetInputCount());
  std::vector<int64_t> device_out_shape = shape0;
  bool can_use_device_sum = ctx.GetInputCount() > 0 && AllGpuInputs(ctx);
  for (size_t input_index = 0; input_index < ctx.GetInputCount();
       ++input_index) {
    auto input = ctx.GetInput(input_index);
    auto input_info = input.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != elem_type) {
      can_use_device_sum = false;
      break;
    }
    input_shapes.push_back(input_info.GetShape());
    device_out_shape = input_index == 0 ? input_shapes.back()
                                        : BroadcastShape(device_out_shape,
                                                         input_shapes.back());
  }

  if (can_use_device_sum) {
    Ort::UnownedValue y = ctx.GetOutput(0, device_out_shape);
    if (IsGpuMemory(y.GetTensorMemoryInfo())) {
      bool handled = false;
      RETURN_IF_ERROR(
          ComputeDeviceSum(ctx, input_shapes, device_out_shape, y, handled,
                           elem_type));
      if (handled) {
        return nullptr;
      }
    }
  }

  return UnsupportedDeviceElementwiseStatus("Sum", elem_type);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sum, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Sum)
