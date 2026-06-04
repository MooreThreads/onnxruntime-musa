// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/op_kernel_common.h"

namespace {
class FusedMatMul : public OpKernelBase<FusedMatMul> {
 public:
  FusedMatMul(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    trans_a_ = AttrOrDefault<int64_t>(kernel_info, "transA", 0);
    trans_b_ = AttrOrDefault<int64_t>(kernel_info, "transB", 0);
    trans_batch_a_ = AttrOrDefault<int64_t>(kernel_info, "transBatchA", 0);
    trans_batch_b_ = AttrOrDefault<int64_t>(kernel_info, "transBatchB", 0);
    alpha_ = AttrOrDefault<float>(kernel_info, "alpha", 1.0f);
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  int64_t trans_a_ = 0;
  int64_t trans_b_ = 0;
  int64_t trans_batch_a_ = 0;
  int64_t trans_batch_b_ = 0;
  float alpha_ = 1.0f;
};

OrtStatus* FusedMatMul::Compute(Ort::KernelContext& ctx) const {
  bool trans_a = trans_a_ != 0;
  bool trans_b = trans_b_ != 0;
  if (trans_batch_a_ != 0 || trans_batch_b_ != 0) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "FusedMatMul transBatch is not implemented");
  }
  auto a_info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto b_info = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  if (a_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "FusedMatMul only supports float tensors");
  }

  std::vector<int64_t> a_shape = a_info.GetShape();
  std::vector<int64_t> b_shape = b_info.GetShape();
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "FusedMatMul requires rank >= 2 inputs");
  }

  int64_t m =
      trans_a ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
  int64_t k =
      trans_a ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
  int64_t kb =
      trans_b ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
  int64_t n =
      trans_b ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  if (k != kb) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "FusedMatMul K dimension mismatch");
  }

  std::vector<int64_t> a_batch = PrefixShape(a_shape, 2);
  std::vector<int64_t> b_batch = PrefixShape(b_shape, 2);
  std::vector<int64_t> batch_shape = BroadcastShape(a_batch, b_batch);
  std::vector<int64_t> out_shape = batch_shape;
  out_shape.push_back(m);
  out_shape.push_back(n);

  std::vector<float> a = ReadTyped<float>(ctx.GetInput(0));
  std::vector<float> b = ReadTyped<float>(ctx.GetInput(1));
  std::vector<float> out(static_cast<size_t>(NumElements(out_shape)), 0.0f);
  auto a_strides = Strides(a_shape);
  auto b_strides = Strides(b_shape);
  auto out_strides = Strides(out_shape);
  int64_t batch_total = NumElements(batch_shape);

  for (int64_t batch_idx = 0; batch_idx < batch_total; ++batch_idx) {
    std::vector<int64_t> batch_coord = Coordinates(batch_idx, batch_shape);
    std::vector<int64_t> a_batch_coord =
        BroadcastBatchCoord(batch_coord, batch_shape, a_batch);
    std::vector<int64_t> b_batch_coord =
        BroadcastBatchCoord(batch_coord, batch_shape, b_batch);
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int64_t kk = 0; kk < k; ++kk) {
          std::vector<int64_t> a_coord = a_batch_coord;
          std::vector<int64_t> b_coord = b_batch_coord;
          if (trans_a) {
            a_coord.push_back(kk);
            a_coord.push_back(row);
          } else {
            a_coord.push_back(row);
            a_coord.push_back(kk);
          }
          if (trans_b) {
            b_coord.push_back(col);
            b_coord.push_back(kk);
          } else {
            b_coord.push_back(kk);
            b_coord.push_back(col);
          }
          sum += a[static_cast<size_t>(Offset(a_coord, a_strides))] *
                 b[static_cast<size_t>(Offset(b_coord, b_strides))];
        }
        std::vector<int64_t> out_coord = batch_coord;
        out_coord.push_back(row);
        out_coord.push_back(col);
        out[static_cast<size_t>(Offset(out_coord, out_strides))] = alpha_ * sum;
      }
    }
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  return WriteTyped<float>(y, out);
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    FusedMatMul, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    FusedMatMul)
