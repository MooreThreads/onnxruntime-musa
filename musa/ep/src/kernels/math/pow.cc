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

bool TryMudnnPow(Ort::KernelContext& ctx, const std::vector<int64_t>& shape0,
                 const std::vector<int64_t>& shape1,
                 ONNXTensorElementDataType lhs_type,
                 ONNXTensorElementDataType rhs_type) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  if (shape0.empty() || shape1.empty() ||
      out_shape.size() > kMudnnMaxElementwiseRank ||
      shape0.size() > kMudnnMaxElementwiseRank ||
      shape1.size() > kMudnnMaxElementwiseRank ||
      !IsGpuMemory(ctx.GetInput(0).GetTensorMemoryInfo()) ||
      !IsGpuMemory(ctx.GetInput(1).GetTensorMemoryInfo())) {
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
  if (!SetMudnnTensor(lhs_tensor, ctx.GetInput(0).GetTensorRawData(), shape0,
                      lhs_type) ||
      !SetMudnnTensor(rhs_tensor, ctx.GetInput(1).GetTensorRawData(), shape1,
                      rhs_type) ||
      !SetMudnnTensor(output_tensor, y.GetTensorMutableRawData(), out_shape,
                      lhs_type)) {
    return false;
  }

  ::musa::dnn::Binary op;
  if (op.SetMode(::musa::dnn::Binary::Mode::POW) !=
      ::musa::dnn::Status::SUCCESS) {
    return false;
  }
  return op.Run(*handle, output_tensor, lhs_tensor, rhs_tensor) ==
         ::musa::dnn::Status::SUCCESS;
}

OrtStatus* PowDeviceCompute(Ort::KernelContext& ctx,
                            const std::vector<int64_t>& shape0,
                            const std::vector<int64_t>& shape1,
                            ONNXTensorElementDataType elem_type,
                            ONNXTensorElementDataType rhs_elem_type) {
  std::vector<int64_t> out_shape = BroadcastShape(shape0, shape1);
  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  MusaElementType lhs_musa_elem_type;
  MusaElementType rhs_musa_elem_type;
  if (!ToMusaElementType(elem_type, lhs_musa_elem_type) ||
      !ToMusaElementType(rhs_elem_type, rhs_musa_elem_type) ||
      !IsGpuMemory(lhs.GetTensorMemoryInfo()) ||
      !IsGpuMemory(rhs.GetTensorMemoryInfo()) ||
      !CanUseBroadcastKernel(out_shape, shape0, shape1)) {
    return UnsupportedDeviceElementwiseStatus("Pow", elem_type);
  }

  Ort::UnownedValue y = ctx.GetOutput(0, out_shape);
  if (!IsGpuMemory(y.GetTensorMemoryInfo())) {
    return UnsupportedDeviceElementwiseStatus("Pow", elem_type);
  }

  musaError_t status = LaunchMusaPowKernel(
      lhs.GetTensorRawData(), rhs.GetTensorRawData(),
      y.GetTensorMutableRawData(),
      MakeBroadcastParams(out_shape, shape0, shape1), lhs_musa_elem_type,
      rhs_musa_elem_type, GetComputeStream(ctx));
  if (status == musaErrorNotSupported) {
    return UnsupportedDeviceElementwiseStatus("Pow", elem_type);
  }
  return LaunchStatus(status);
}

class Pow : public OpKernelBase<Pow> {
 public:
  Pow(const OrtKernelInfo* /*info*/, void* /*state*/) {}
  OrtStatus* Compute(Ort::KernelContext& ctx) const;
};

OrtStatus* Pow::Compute(Ort::KernelContext& ctx) const {
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape0 = info.GetShape();
  auto rhs_info = ctx.GetInput(1).GetTensorTypeAndShapeInfo();
  auto rhs_elem_type = rhs_info.GetElementType();
  auto shape1 = rhs_info.GetShape();
  if (TryMudnnPow(ctx, shape0, shape1, elem_type, rhs_elem_type)) {
    return nullptr;
  }
  if (elem_type != rhs_elem_type) {
    return PowDeviceCompute(ctx, shape0, shape1, elem_type, rhs_elem_type);
  }
  return BinaryDeviceCompute(ctx, shape0, shape1, elem_type, MusaBinaryOp::Pow,
                             "Pow");
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pow, kOnnxDomain, 13, 14,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", PowTensorTypes())
         .AddTypeConstraint("T1", PowExponentOpset13TensorTypes())),
    Pow)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Pow, kOnnxDomain, 15, 19,
                                  (Ort::KernelDefBuilder()
                                       .AddTypeConstraint("T", PowTensorTypes())
                                       .AddTypeConstraint("T1",
                                                          PowTensorTypes())),
                                  Pow)
