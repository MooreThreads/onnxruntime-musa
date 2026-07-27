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

#include "math/mod_impl.h"
#include "shared_inc/op_kernel_common.h"

namespace {

class Mod : public OpKernelBase<Mod> {
 public:
  Mod(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    fmod_ = AttrOrDefault<int64_t>(kernel_info, "fmod", 0) != 0;
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  bool fmod_ = false;
};

OrtStatus* Mod::Compute(Ort::KernelContext& ctx) const {
  if (fmod_) {
    return Ort::GetApi().CreateStatus(
        ORT_NOT_IMPLEMENTED, "Mod MUSA only supports integer mod fmod=0");
  }

  Ort::ConstValue lhs = ctx.GetInput(0);
  Ort::ConstValue rhs = ctx.GetInput(1);
  auto lhs_info = lhs.GetTensorTypeAndShapeInfo();
  auto rhs_info = rhs.GetTensorTypeAndShapeInfo();
  auto elem_type = lhs_info.GetElementType();
  if (rhs_info.GetElementType() != elem_type) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "Mod input dtypes must match");
  }
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Mod MUSA only supports int32/int64");
  }

  auto lhs_shape = lhs_info.GetShape();
  auto rhs_shape = rhs_info.GetShape();
  auto output_shape = BroadcastShape(lhs_shape, rhs_shape);
  if (!CanUseBroadcastKernel(output_shape, lhs_shape, rhs_shape)) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Mod rank exceeds MUSA device limit");
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
  if (!IsGpuMemory(output.GetTensorMemoryInfo())) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Mod requires MUSA output");
  }

  musaStream_t stream = GetComputeStream(ctx);
  DeviceInputBuffer lhs_buffer;
  DeviceInputBuffer rhs_buffer;
  RETURN_IF_ERROR(lhs_buffer.Bind(lhs, stream));
  RETURN_IF_ERROR(rhs_buffer.Bind(rhs, stream));

  musaError_t status = LaunchMusaModKernel(
      lhs_buffer.data(), rhs_buffer.data(), output.GetTensorMutableRawData(),
      MakeBroadcastParams(output_shape, lhs_shape, rhs_shape),
      static_cast<int32_t>(elem_type), stream);
  if (status == musaErrorNotSupported) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED,
                                      "Mod MUSA only supports int32/int64");
  }
  return LaunchStatus(status);
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mod, kOnnxDomain, 10, 12,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", IntTensorTypes())), Mod)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mod, kOnnxDomain, 13, 19,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", IntTensorTypes())), Mod)
