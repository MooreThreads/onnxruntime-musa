// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
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
        static_cast<int32_t>(elem_size), params, nullptr));
  }
  std::vector<uint8_t> in;
  RETURN_IF_ERROR(CopyToHost(input0, in));
  std::vector<uint8_t> out(static_cast<size_t>(NumElements(out_shape)) *
                           elem_size);
  auto in_strides = Strides(shape0);
  for (int64_t i = 0; i < NumElements(out_shape); ++i) {
    auto out_coord = Coordinates(i, out_shape);
    std::vector<int64_t> in_coord(shape0.size());
    for (size_t j = 0; j < perm.size(); ++j)
      in_coord[static_cast<size_t>(perm[j])] = out_coord[j];
    std::memcpy(out.data() + static_cast<size_t>(i) * elem_size,
                in.data() + static_cast<size_t>(Offset(in_coord, in_strides)) *
                                elem_size,
                elem_size);
  }
  return CopyFromHost(y, out.data(), out.size());
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", TensorTypesWithBool())),
    Transpose)
