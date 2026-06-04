// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"
#include "tensor/gather_impl.h"

namespace {
class Gather : public OpKernelBase<Gather> {
 public:
  Gather(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    axis_ = AttrOrDefault<int64_t>(kernel_info, "axis", 0);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t axis_ = 0;
};

OrtStatus* Gather::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input0 = ctx.GetInput(0);
  auto in0_info = input0.GetTensorTypeAndShapeInfo();
  auto elem_type = in0_info.GetElementType();
  auto shape0 = in0_info.GetShape();
  int64_t axis = NormalizeAxis(axis_, shape0.size());
  Ort::ConstValue indices_value = ctx.GetInput(1);
  auto indices_info = indices_value.GetTensorTypeAndShapeInfo();
  auto indices_shape = indices_info.GetShape();
  std::vector<int64_t> out_shape;
  out_shape.insert(out_shape.end(), shape0.begin(), shape0.begin() + axis);
  out_shape.insert(out_shape.end(), indices_shape.begin(), indices_shape.end());
  out_shape.insert(out_shape.end(), shape0.begin() + axis + 1, shape0.end());
  size_t elem_size = ElementSize(elem_type);
  if (elem_size == 0)
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gather unsupported dtype");
  size_t index_elem_size = ElementSize(indices_info.GetElementType());
  if (index_elem_size != sizeof(int32_t) &&
      index_elem_size != sizeof(int64_t)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Gather unsupported index dtype");
  }
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (IsGpuMemory(input0.GetTensorMemoryInfo()) &&
      IsGpuMemory(indices_value.GetTensorMemoryInfo()) &&
      IsGpuMemory(y.GetTensorMemoryInfo())) {
    int64_t block_size = 1;
    for (size_t dim = static_cast<size_t>(axis) + 1; dim < shape0.size(); ++dim)
      block_size *= shape0[dim];
    int64_t indices_count = NumElements(indices_shape);
    int64_t indices_max = shape0[static_cast<size_t>(axis)];
    int64_t prefix_count = 1;
    for (int64_t dim = 0; dim < axis; ++dim)
      prefix_count *= shape0[static_cast<size_t>(dim)];
    int64_t input_block_size = indices_max * block_size;
    int64_t output_block_size = indices_count * block_size;
    return LaunchStatus(LaunchMusaGatherKernel(
        input0.GetTensorRawData(), indices_value.GetTensorRawData(),
        y.GetTensorMutableRawData(), static_cast<int32_t>(elem_size),
        static_cast<int32_t>(index_elem_size), input_block_size, indices_max,
        output_block_size, block_size, prefix_count * output_block_size,
        nullptr));
  }

  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                    "Gather requires MUSA input, indices, and output");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", AllFixedSizeTensorTypes())
         .AddTypeConstraint("Tind", IntTensorTypes())),
    Gather)
