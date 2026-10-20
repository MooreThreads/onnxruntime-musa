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

#include "activation/gelu_support.h"
#include "shared_inc/op_kernel_common.h"

namespace {
class Gelu : public OpKernelBase<Gelu> {
 public:
  Gelu(const OrtKernelInfo* info, void* /*state*/) {
    Ort::ConstKernelInfo kernel_info(info);
    // The ONNX Gelu (opset 20) declares `approximate` as a string, the
    // com.microsoft Gelu as an int (0 = exact, 1 = tanh); read whichever the
    // schema of the matched domain carries.
    try {
      approximate_ = kernel_info.GetAttribute<std::string>("approximate");
    } catch (...) {
      try {
        approximate_ = kernel_info.GetAttribute<int64_t>("approximate") == 0
                           ? musa_ep::kGeluDefaultApproximate
                           : "tanh";
      } catch (...) {
        approximate_ = musa_ep::kGeluDefaultApproximate;
      }
    }
  }
  OrtStatus* Compute(Ort::KernelContext& ctx) const;

 private:
  std::string approximate_;
};

OrtStatus* Gelu::Compute(Ort::KernelContext& ctx) const {
  if (const char* reason = musa_ep::GeluApproximateRejection(approximate_);
      reason != nullptr) {
    return Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, reason);
  }
  auto info = ctx.GetInput(0).GetTensorTypeAndShapeInfo();
  auto elem_type = info.GetElementType();
  auto shape = info.GetShape();
  if (OutputEmptyTensorIfNeeded(ctx, shape)) {
    return nullptr;
  }
  return UnaryDeviceCompute(ctx, shape, elem_type, MusaUnaryOp::Gelu, "Gelu");
}
}  // namespace

// Gelu became an official ONNX operator at opset 20; the same kernel class also
// answers the com.microsoft domain, whose Gelu has the same semantics and the
// same attribute.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gelu, kOnnxDomain, 20, 21,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Gelu)

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gelu, kMSDomain, 1, 1,
    (Ort::KernelDefBuilder().AddTypeConstraint("T", FloatLikeTensorTypes())),
    Gelu)
