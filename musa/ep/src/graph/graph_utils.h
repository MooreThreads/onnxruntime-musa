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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace musa_ep {

bool IsOnnxDomain(const std::string& domain);
bool IsOnnxOp(const Ort::ConstNode& node, const char* op_type);

std::string Name(Ort::ConstValueInfo value_info);

bool IsFloatTensorValueInfo(Ort::ConstValueInfo value_info);
bool IsConstantInitializerValueInfo(Ort::ConstValueInfo value_info);
bool IsIntTensorValueInfo(Ort::ConstValueInfo value_info);
bool IsSmallInitializer(Ort::ConstValueInfo input);

std::optional<std::vector<int64_t>> GetStaticShape(
    Ort::ConstValueInfo value_info);
std::optional<int64_t> GetIntAttribute(Ort::ConstNode node,
                                       const std::string& name);
std::optional<std::vector<int64_t>> GetIntsAttribute(Ort::ConstNode node,
                                                     const std::string& name);
std::optional<std::string> GetStringAttribute(Ort::ConstNode node,
                                              const std::string& name);

bool NormalizeAxis(int64_t axis, size_t rank, int64_t& normalized_axis);
bool KnownDimsEqual(int64_t lhs, int64_t rhs);
bool ShapesEqualOnKnownDims(const std::vector<int64_t>& lhs,
                            const std::vector<int64_t>& rhs);

constexpr int64_t kSmallInitializerThreshold = 100;

std::optional<std::vector<int64_t>> ReadSmallIntInitializer(
    Ort::ConstValueInfo value_info);
std::optional<std::vector<int64_t>> ReadIntInitializerNoLimit(
    Ort::ConstValueInfo value_info);
std::optional<float> ReadScalarFloatInitializer(Ort::ConstValueInfo value_info);
std::optional<std::vector<int64_t>> ReadUnsqueezeAxes(
    Ort::ConstNode unsqueeze_node);

bool IsZeroFloatConstantOfShape(Ort::ConstNode node);
std::optional<std::vector<int64_t>> ConstantOfShapeOutputShape(
    Ort::ConstNode node, Ort::ConstValueInfo output_info);

}  // namespace musa_ep
