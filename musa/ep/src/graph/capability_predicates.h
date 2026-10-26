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

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace musa_ep {

/// Graph-level capability gate for a single node that already matched a
/// registered kernel.
///
/// `OrtEpApi::EpGraphSupportInfo_LookUpKernel` matches op/domain/opset/dtype
/// only, so an operator whose kernel implements a subset of the ONNX
/// configurations (such as Resize, where the kernels only cover nearest/
/// asymmetric/floor and the linear coordinate transformations, rank 4 and 5,
/// and an empty roi) would otherwise be handed every node of that operator and
/// only fail once the kernel runs. Returning false here leaves the node to the
/// CPU EP, as it was before the kernel was registered.
///
/// The per-operator predicates live next to the kernels they gate so that the
/// capability decision and the kernel's runtime guard are two uses of one
/// definition instead of two copies that drift apart.
bool IsNodeConfigurationSupported(Ort::ConstNode node);

}  // namespace musa_ep
