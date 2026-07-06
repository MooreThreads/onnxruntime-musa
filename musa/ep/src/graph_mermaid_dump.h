// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

Ort::Status DumpGetCapabilityGraphToMermaidIfEnabled(const OrtGraph& ort_graph);
