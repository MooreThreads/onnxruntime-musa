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

#include "math/einsum_impl.h"
#include "math/matmul.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class Einsum : public OpKernelBase<Einsum> {
 public:
  Einsum(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    equation_ = AttrOrDefault<std::string>(kernel_info, "equation", "");
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  OrtStatus* ComputeDiagonal(Ort::KernelContext& ctx) const;
  OrtStatus* ComputeBhijHk(Ort::KernelContext& ctx) const;
  OrtStatus* ComputeIjBjk(Ort::KernelContext& ctx) const;
  OrtStatus* ComputeBlhwBjhwBhl(Ort::KernelContext& ctx) const;
  OrtStatus* ComputeIlhwBjhwBhl(Ort::KernelContext& ctx) const;
  OrtStatus* ComputeNikBnkBni(Ort::KernelContext& ctx) const;
  OrtStatus* ComputeBnkNkdBnd(Ort::KernelContext& ctx) const;

  std::string equation_;
};

OrtStatus* Einsum::ComputeDiagonal(Ort::KernelContext& ctx) const {
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 2 || shape[0] != shape[1]) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum aa->a requires square rank-2 input");
  }
  const size_t elem_size = ElementSize(info.GetElementType());
  if (elem_size == 0) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum unsupported dtype");
  }
  if (!IsGpuMemory(input.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA input");
  }
  Ort::UnownedValue output = ctx.GetOutput(0, {shape[0]});
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }
  return LaunchStatus(LaunchMusaEinsumDiagonalKernel(
      input.GetTensorRawData(), output.GetTensorMutableRawData(), shape[0],
      static_cast<int32_t>(elem_size), GetComputeStream(ctx)));
}

OrtStatus* Einsum::ComputeBhijHk(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum bhij,hk->bkij only supports float");
  }
  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  if (lhs_shape.size() != 4 || rhs_shape.size() != 2 ||
      lhs_shape[1] != rhs_shape[0]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum bhij,hk->bkij shape mismatch");
  }
  if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA inputs");
  }

  std::vector<int64_t> output_shape = {lhs_shape[0], rhs_shape[1], lhs_shape[2],
                                       lhs_shape[3]};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }
  return LaunchStatus(LaunchMusaEinsumBhijHkKernel(
      lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), lhs_shape[0], lhs_shape[1],
      rhs_shape[1], lhs_shape[2], lhs_shape[3], GetComputeStream(ctx)));
}

OrtStatus* Einsum::ComputeIjBjk(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum ij,bjk->bik only supports float");
  }
  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  if (lhs_shape.size() != 2 || rhs_shape.size() != 3 ||
      lhs_shape[1] != rhs_shape[1]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum ij,bjk->bik shape mismatch");
  }
  if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA inputs");
  }

  std::vector<int64_t> output_shape = {rhs_shape[0], lhs_shape[0],
                                       rhs_shape[2]};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }

  const std::vector<int64_t> matmul_lhs_shape = {1, lhs_shape[0], lhs_shape[1]};
  return ComputeMusaMatMulDevice(
      lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      matmul_lhs_shape, rhs_shape, output_shape, false, false, false, false,
      1.0f, GetComputeStream(ctx));
}

OrtStatus* Einsum::ComputeBlhwBjhwBhl(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum blhw,bjhw->bhl only supports float");
  }
  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  if (lhs_shape.size() != 4 || rhs_shape.size() != 4 ||
      lhs_shape[0] != rhs_shape[0] || lhs_shape[2] != rhs_shape[2] ||
      lhs_shape[3] != rhs_shape[3]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum blhw,bjhw->bhl shape mismatch");
  }
  if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA inputs");
  }

  std::vector<int64_t> output_shape = {lhs_shape[0], lhs_shape[2],
                                       lhs_shape[1]};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }

  return LaunchStatus(LaunchMusaEinsumBlhwBjhwBhlKernel(
      lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), lhs_shape[0], lhs_shape[1],
      lhs_shape[2], lhs_shape[3], rhs_shape[1], GetComputeStream(ctx)));
}

OrtStatus* Einsum::ComputeIlhwBjhwBhl(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum ilhw,bjhw->bhl only supports float");
  }
  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  if (lhs_shape.size() != 4 || rhs_shape.size() != 4 ||
      lhs_shape[2] != rhs_shape[2] || lhs_shape[3] != rhs_shape[3]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum ilhw,bjhw->bhl shape mismatch");
  }
  if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA inputs");
  }

  std::vector<int64_t> output_shape = {rhs_shape[0], lhs_shape[2],
                                       lhs_shape[1]};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }

  return LaunchStatus(LaunchMusaEinsumIlhwBjhwBhlKernel(
      lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), rhs_shape[0], lhs_shape[0],
      lhs_shape[1], lhs_shape[2], lhs_shape[3], rhs_shape[1],
      GetComputeStream(ctx)));
}

OrtStatus* Einsum::ComputeNikBnkBni(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum nik,bnk->bni only supports float");
  }
  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  if (lhs_shape.size() != 3 || rhs_shape.size() != 3 ||
      lhs_shape[0] != rhs_shape[1] || lhs_shape[2] != rhs_shape[2]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum nik,bnk->bni shape mismatch");
  }
  if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA inputs");
  }

  std::vector<int64_t> output_shape = {rhs_shape[0], lhs_shape[0],
                                       lhs_shape[1]};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }

  return LaunchStatus(LaunchMusaEinsumNikBnkBniKernel(
      lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), rhs_shape[0], lhs_shape[0],
      lhs_shape[1], lhs_shape[2], GetComputeStream(ctx)));
}

OrtStatus* Einsum::ComputeBnkNkdBnd(Ort::KernelContext& ctx) const {
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  if (lhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Einsum bnk,nkd->bnd only supports float");
  }
  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  if (lhs_shape.size() != 3 || rhs_shape.size() != 3 ||
      lhs_shape[1] != rhs_shape[0] || lhs_shape[2] != rhs_shape[1]) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum bnk,nkd->bnd shape mismatch");
  }
  if (!IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA inputs");
  }

  std::vector<int64_t> output_shape = {lhs_shape[0], lhs_shape[1],
                                       rhs_shape[2]};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Einsum requires MUSA output");
  }

  return LaunchStatus(LaunchMusaEinsumBnkNkdBndKernel(
      lhs.GetTensorData<float>(), rhs.GetTensorData<float>(),
      output.GetTensorMutableData<float>(), lhs_shape[0], lhs_shape[1],
      lhs_shape[2], rhs_shape[2], GetComputeStream(ctx)));
}

OrtStatus* Einsum::Compute(Ort::KernelContext& ctx) const {
  if (equation_ == "aa->a") {
    return ComputeDiagonal(ctx);
  }
  if (equation_ == "bhij,hk->bkij") {
    return ComputeBhijHk(ctx);
  }
  if (equation_ == "ij,bjk->bik") {
    return ComputeIjBjk(ctx);
  }
  if (equation_ == "blhw,bjhw->bhl") {
    return ComputeBlhwBjhwBhl(ctx);
  }
  if (equation_ == "ilhw,bjhw->bhl") {
    return ComputeIlhwBjhwBhl(ctx);
  }
  if (equation_ == "nik,bnk->bni") {
    return ComputeNikBnkBni(ctx);
  }
  if (equation_ == "bnk,nkd->bnd") {
    return ComputeBnkNkdBnd(ctx);
  }
  const std::string message =
      "unsupported Einsum equation for MUSA device path: " + equation_;
  return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, message.c_str());
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Einsum, kOnnxDomain, 12, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatTensorTypes())),
    Einsum)
