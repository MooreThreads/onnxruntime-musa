// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace musa_ep {

enum class FusionFloatPolicy {
  kNoFloatGate,
  kFp32Only,
  kFp32Fp16Bf16,
  kFloat64Only,
  kAnyFloat,
};

struct FusionDTypeContract {
  const char* storage_types;
  const char* accumulator_type;
  const char* output_type_policy;
  const char* cast_policy;
  FusionFloatPolicy float_policy;
};

struct FusionComputeConfig {
  std::string storage_types;
  std::string accumulator_type;
  std::string output_type_policy;
  std::string cast_policy;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<ONNXTensorElementDataType> output_types;
};

struct FusionFloat32Tag {
  using StorageType = float;
  static constexpr ONNXTensorElementDataType element_type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

struct FusionFloat16Tag {
  using StorageType = uint16_t;
  static constexpr ONNXTensorElementDataType element_type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};

struct FusionBFloat16Tag {
  using StorageType = uint16_t;
  static constexpr ONNXTensorElementDataType element_type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
};

struct FusionFloat64Tag {
  using StorageType = double;
  static constexpr ONNXTensorElementDataType element_type =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
};

template <typename Fn>
auto DispatchByElementType(ONNXTensorElementDataType elem_type, Fn&& fn)
    -> decltype(fn.template operator()<FusionFloat32Tag>()) {
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return fn.template operator()<FusionFloat32Tag>();
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return fn.template operator()<FusionFloat16Tag>();
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return fn.template operator()<FusionBFloat16Tag>();
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return fn.template operator()<FusionFloat64Tag>();
    default:
      throw std::runtime_error("unsupported fusion dispatch dtype");
  }
}

const FusionDTypeContract& FusionDTypeContractForFinder(const char* finder);
std::optional<ONNXTensorElementDataType> GetTensorElementType(
    Ort::ConstValueInfo value_info);
const char* FusionDTypeName(ONNXTensorElementDataType type);
bool IsFloatingStorageType(ONNXTensorElementDataType type);
bool IsSupportedStorageType(ONNXTensorElementDataType type,
                            const FusionDTypeContract& contract);
bool RequireSameElementType(const std::vector<Ort::ConstValueInfo>& value_infos,
                            ONNXTensorElementDataType& elem_type,
                            std::string* reason = nullptr);

}  // namespace musa_ep
