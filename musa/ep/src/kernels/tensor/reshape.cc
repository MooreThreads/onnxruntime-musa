// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class Reshape : public OpKernelBase<Reshape> {
 public:
  Reshape(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    allowzero_ = AttrOrDefault<int64_t>(kernel_info, "allowzero", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t allowzero_ = 0;
};

class ReshapeString : public OpKernelBase<ReshapeString> {
 public:
  ReshapeString(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    allowzero_ = AttrOrDefault<int64_t>(kernel_info, "allowzero", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t allowzero_ = 0;
};

std::vector<int64_t> ResolveReshapeOutputShape(
    const std::vector<int64_t>& input_shape, std::vector<int64_t> out_shape,
    int64_t allowzero) {
  int64_t input_size = NumElements(input_shape);
  int64_t known = 1;
  int infer_idx = -1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] == 0 && !allowzero) out_shape[i] = input_shape[i];
    if (out_shape[i] == -1) {
      infer_idx = static_cast<int>(i);
    } else {
      known *= out_shape[i];
    }
  }
  if (infer_idx >= 0)
    out_shape[static_cast<size_t>(infer_idx)] = input_size / known;
  return out_shape;
}

OrtStatus* Reshape::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto shape0 = input0.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> out_shape =
      ResolveReshapeOutputShape(shape0, ReadIntTensor(ctx, 1), allowzero_);
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return CopyRawTensor(input0, y, input0.GetTensorSizeInBytes(),
                       GetComputeStream(ctx));
}

OrtStatus* ReshapeString::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto input_info = input0.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "ReshapeString requires string input");
  }
  std::vector<int64_t> out_shape = ResolveReshapeOutputShape(
      input_info.GetShape(), ReadIntTensor(ctx, 1), allowzero_);
  Ort::UnownedValue output = ctx.GetOutput(0, out_shape);
  if (IsGpuMemory(input0.GetTensorMemoryInfo()) ||
      IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "ReshapeString uses CPU memory because ONNX string tensors have no "
        "MUSA device representation");
  }

  const size_t count = static_cast<size_t>(input_info.GetElementCount());
  std::vector<std::string> values;
  values.reserve(count);
  std::vector<const char*> raw;
  raw.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(input0.GetStringTensorElement(i));
    raw.push_back(values.back().c_str());
  }
  output.FillStringTensor(raw.data(), raw.size());
  return nullptr;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .SetInputMemType(1, OrtMemTypeCPUInput)),
    Reshape)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape, kOnnxDomain, 19, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", StringTensorTypes())
         .SetInputMemType(0, OrtMemTypeCPUInput)
         .SetInputMemType(1, OrtMemTypeCPUInput)
         .SetOutputMemType(0, OrtMemTypeCPUOutput)),
    ReshapeString)
