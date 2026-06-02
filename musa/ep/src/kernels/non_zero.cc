// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/op_kernel_common.h"

namespace {
class NonZero : public OpKernelBase<NonZero> {
 public:
  NonZero(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* NonZero::Compute(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  int64_t rank = static_cast<int64_t>(shape.size());
  int64_t total = NumElements(shape);

  std::vector<uint8_t> raw;
  RETURN_IF_ERROR(CopyToHost(input, raw));
  size_t elem_size = raw.size() / static_cast<size_t>(total);

  // Collect linear indices of nonzero elements (row-major / C order)
  std::vector<int64_t> nz_linear;
  for (int64_t i = 0; i < total; ++i) {
    const uint8_t* ptr = raw.data() + static_cast<size_t>(i) * elem_size;
    bool nonzero = false;
    for (size_t b = 0; b < elem_size; ++b) nonzero |= (ptr[b] != 0);
    if (nonzero) nz_linear.push_back(i);
  }

  int64_t num_nz = static_cast<int64_t>(nz_linear.size());
  // Output shape: [rank, num_nz]
  std::vector<int64_t> out_shape = {rank, num_nz};
  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);

  if (num_nz == 0) return nullptr;

  // Build output: column i = multi-index of nz_linear[i]
  std::vector<int64_t> out_data(static_cast<size_t>(rank * num_nz));
  for (int64_t j = 0; j < num_nz; ++j) {
    auto coord = Coordinates(nz_linear[static_cast<size_t>(j)], shape);
    for (int64_t d = 0; d < rank; ++d)
      out_data[static_cast<size_t>(d * num_nz + j)] =
          coord[static_cast<size_t>(d)];
  }
  return WriteTyped<int64_t>(y, out_data);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    NonZero, kOnnxDomain, 13, 17,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", TensorTypesWithBool())),
    NonZero)
