// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

#include "shared_inc/blas_utils.h"
#include "shared_inc/op_kernel_common.h"

namespace {
constexpr size_t kMudnnMaxElementwiseRank = 5;

bool TryMudnnEqual(Ort::KernelContext& ctx, const std::vector<int64_t>& shape0,
                   const std::vector<int64_t>& shape1,
                   ONNXTensorElementDataType elem_type) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  if (out_shape.size() > kMudnnMaxElementwiseRank ||
      shape0.size() > kMudnnMaxElementwiseRank ||
      shape1.size() > kMudnnMaxElementwiseRank ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return false;
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return false;
  }

  ::musa::dnn::Handle* handle = nullptr;
  OrtStatus* handle_status = EnsureMudnnHandle(&handle, GetComputeStream(ctx));
  if (handle_status != nullptr) {
    Ort::GetApi().ReleaseStatus(handle_status);
    return false;
  }

  ::musa::dnn::Tensor lhs_tensor;
  ::musa::dnn::Tensor rhs_tensor;
  ::musa::dnn::Tensor output_tensor;
  if (!SetMudnnTensor(lhs_tensor, lhs.GetTensorRawData(), shape0, elem_type) ||
      !SetMudnnTensor(rhs_tensor, rhs.GetTensorRawData(), shape1, elem_type) ||
      !SetMudnnTensor(output_tensor, y.GetTensorMutableRawData(), out_shape,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)) {
    return false;
  }

  ::musa::dnn::Binary op;
  if (op.SetMode(::musa::dnn::Binary::Mode::EQ) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

class Equal : public OpKernelBase<Equal> {
 public:
  Equal(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

class EqualString : public OpKernelBase<EqualString> {
 public:
  EqualString(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

std::vector<std::string_view> ReadStringTensorViews(
    Ort::ConstValue value, std::vector<char>& buffer,
    std::vector<size_t>& offsets) {
  auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = static_cast<size_t>(NumElements(info.GetShape()));
  const size_t data_length = value.GetStringTensorDataLength();
  buffer.resize(data_length);
  offsets.resize(count);
  if (count > 0) {
    value.GetStringTensorContent(buffer.data(), buffer.size(), offsets.data(),
                                 offsets.size());
  }

  std::vector<std::string_view> views;
  views.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const size_t begin = offsets[i];
    const size_t end = i + 1 < count ? offsets[i + 1] : buffer.size();
    views.emplace_back(buffer.data() + begin, end - begin);
  }
  return views;
}

OrtStatus* Equal::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto shape1 = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  if (TryMudnnEqual(ctx, shape0, shape1, elem_type)) {
    return nullptr;
  }
  return CompareDeviceCompute(ctx, shape0, shape1, elem_type,
                              MusaCompareOp::Equal, "Equal");
}

OrtStatus* EqualString::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "EqualString requires string inputs");
  }

  const std::vector<int64_t> lhs_shape = lhs_info.GetShape();
  const std::vector<int64_t> rhs_shape = rhs_info.GetShape();
  const std::vector<int64_t> out_shape = BroadcastShape(lhs_shape, rhs_shape);
  Ort::UnownedValue output = ctx.GetOutput(0, out_shape);
  if (IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "EqualString requires CPU string inputs and MUSA bool output");
  }

  const int64_t total = NumElements(out_shape);
  const std::vector<int64_t> lhs_strides = Strides(lhs_shape);
  const std::vector<int64_t> rhs_strides = Strides(rhs_shape);
  std::vector<char> lhs_buffer;
  std::vector<char> rhs_buffer;
  std::vector<size_t> lhs_offsets;
  std::vector<size_t> rhs_offsets;
  std::vector<std::string_view> lhs_values =
      ReadStringTensorViews(lhs, lhs_buffer, lhs_offsets);
  std::vector<std::string_view> rhs_values =
      ReadStringTensorViews(rhs, rhs_buffer, rhs_offsets);
  std::vector<uint8_t> out(static_cast<size_t>(total));
  bool uniform_output = true;
  uint8_t first_value = 0;
  for (int64_t i = 0; i < total; ++i) {
    const std::vector<int64_t> coord = Coordinates(i, out_shape);
    const int64_t lhs_offset = BroadcastOffset(coord, lhs_shape, lhs_strides);
    const int64_t rhs_offset = BroadcastOffset(coord, rhs_shape, rhs_strides);
    const uint8_t value =
        static_cast<uint8_t>(lhs_values[static_cast<size_t>(lhs_offset)] ==
                             rhs_values[static_cast<size_t>(rhs_offset)]);
    out[static_cast<size_t>(i)] = value;
    if (i == 0) {
      first_value = value;
    } else if (value != first_value) {
      uniform_output = false;
    }
  }
  if (uniform_output) {
    musaError_t status =
        musaMemsetAsync(output.GetTensorMutableRawData(), first_value,
                        out.size(), GetComputeStream(ctx));
    if (status != musaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MusaErrorString(status));
    }
    return nullptr;
  }
  return CopyFromHost(output, out.data(), out.size(), GetComputeStream(ctx));
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Equal, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", EqualTensorTypes())), Equal)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Equal, kOnnxDomain, 19, 19,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T",
                                                          StringTensorTypes())
                                       .SetInputMemType(0, OrtMemTypeCPUInput)
                                       .SetInputMemType(1, OrtMemTypeCPUInput)),
                                  EqualString)
