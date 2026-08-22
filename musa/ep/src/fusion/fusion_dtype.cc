// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "fusion/fusion_dtype.h"

#include <stdexcept>

namespace musa_ep {
namespace {

constexpr FusionDTypeContract kNoFusionDTypeContract = {
    "none", "none", "reject", "not-declared", FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kFp32OnlyDTypeContract = {
    "float32", "float32", "preserve graph output dtype",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kFp32Only};

constexpr FusionDTypeContract kFloatIdentityDTypeContract = {
    "all fixed-size tensor storage", "none", "preserve graph output dtype",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kFloatIdentityWithFp32ReduceDTypeContract = {
    "all fixed-size tensor storage; Sum outputs remain float32",
    "float32 for Sum terms", "preserve graph output dtype",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kIntegerDTypeContract = {
    "int32/int64", "none", "preserve graph output dtype",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kEmbeddingGatherDTypeContract = {
    "index tensors plus fixed-size value tensor storage", "none",
    "preserve graph output dtype", "Cast nodes are not part of this fusion",
    FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kIntegerMaskDTypeContract = {
    "int32/int64 ids; bool/int32/float32 mask outputs where implemented",
    "none", "preserve graph output dtype",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kGenerateMTGRCustomMaskDTypeContract = {
    "shape-only tensor inputs of any dtype; bool/int32 mask and int64 "
    "metadata outputs",
    "none", "preserve matched graph output dtype",
    "CroppedInt32 absorbs the final bool-to-int32 Cast",
    FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kMhtaSdpaDTypeContract = {
    "float32/float16/bfloat16 Q/K/V with bool or absorbed int32 keep-mask; "
    "float32/float16/bfloat16/float64 additive-mask path",
    "float32 workspace", "same as Q/K/V",
    "absorbs the UniRank Slice/Cast/Equal int32 keep-mask tail where matched",
    FusionFloatPolicy::kAnyFloat};

constexpr FusionDTypeContract kAttentionProjectionDTypeContract = {
    "float32", "float32",
    "preserve graph output dtype; projection remains FP32-only",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kFp32Only};

constexpr FusionDTypeContract kRmsNormDTypeContract = {
    "float32/float16/bfloat16/float64",
    "float32 internal reduction for all storage types",
    "preserve graph output dtype; gamma and epsilon must match input storage "
    "dtype",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kAnyFloat};

constexpr FusionDTypeContract kCastRmsNormDTypeContract = {
    "bfloat16 input/gamma/output with float32 normalization intermediates",
    "float32 internal reduction",
    "preserve graph output dtype after explicit fp32->bfloat16 rounding",
    "explicit bfloat16->float32 entry Cast and float32->bfloat16 exit Cast",
    FusionFloatPolicy::kFp32Fp16Bf16};

constexpr FusionDTypeContract kFloatReduceDTypeContract = {
    "float32", "float32 reduction", "preserve graph output dtype",
    "Cast nodes are not part of this fusion", FusionFloatPolicy::kFp32Only};

constexpr FusionDTypeContract kRecRankDTypeContract = {
    "float64 inputs, float32 output", "double->float32",
    "preserve graph output dtype", "Cast nodes are not part of this fusion",
    FusionFloatPolicy::kNoFloatGate};

constexpr FusionDTypeContract kLinearGemmDTypeContract = {
    "float32/float16/bfloat16/float64", "float32 for fp16/bfloat16 GEMM",
    "preserve graph output dtype", "Cast nodes are not part of this fusion",
    FusionFloatPolicy::kAnyFloat};

constexpr FusionDTypeContract kFusedMatMulDTypeContract = {
    "float32/float16/bfloat16", "float32 for fp16/bfloat16 GEMM",
    "preserve graph output dtype", "Cast nodes are not part of this fusion",
    FusionFloatPolicy::kFp32Fp16Bf16};

constexpr FusionDTypeContract kMatMulFamilyDTypeContract = {
    "float32/float16/bfloat16", "muDNN/muBLAS FP32 accumulation for FP16/BF16",
    "preserve graph output dtype", "Cast nodes are not part of this fusion",
    FusionFloatPolicy::kFp32Fp16Bf16};

constexpr FusionDTypeContract kSiluDTypeContract = {
    "float32/float16/bfloat16/float64",
    "float32 math for float16/bfloat16; float64 math for float64",
    "same as input", "Cast nodes are not part of this fusion",
    FusionFloatPolicy::kAnyFloat};

bool FloatingTypeAllowed(ONNXTensorElementDataType elem_type,
                         FusionFloatPolicy policy) {
  switch (policy) {
    case FusionFloatPolicy::kNoFloatGate:
      return true;
    case FusionFloatPolicy::kFp32Only:
      return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case FusionFloatPolicy::kFp32Fp16Bf16:
      return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
             elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
             elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case FusionFloatPolicy::kFloat64Only:
      return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case FusionFloatPolicy::kAnyFloat:
      return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
             elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
             elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
             elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  }
  return false;
}

}  // namespace

const FusionDTypeContract& FusionDTypeContractForFinder(const char* finder) {
  const std::string name = finder == nullptr ? "" : finder;
  if (name == "FindMhtaScaledDotProductAttentionFusions") {
    return kMhtaSdpaDTypeContract;
  }
  if (name == "FindGenerateMTGRCustomMaskFusions") {
    return kGenerateMTGRCustomMaskDTypeContract;
  }
  if (name == "FindRecRankCalibrationFusions") {
    return kRecRankDTypeContract;
  }
  if (name == "FindGemmActivationFusions") {
    return kLinearGemmDTypeContract;
  }
  if (name == "FindFusedGemmFusions") {
    return kFusedMatMulDTypeContract;
  }
  if (name == "FindSiluFusions") {
    return kSiluDTypeContract;
  }
  if (name == "FindConcatMatMulFusions" ||
      name == "FindParallelMatMulConcatFusions") {
    return kMatMulFamilyDTypeContract;
  }
  if (name == "FindRmsNormFusions") {
    return kRmsNormDTypeContract;
  }
  if (name == "FindCastRmsNormFusions") {
    return kCastRmsNormDTypeContract;
  }
  if (name == "FindCenteredReduceFusions" || name == "FindSplitReduceFusions" ||
      name == "FindSegmentMaxBroadcastFusions") {
    return kFloatReduceDTypeContract;
  }
  if (name == "FindMultiKqvMhaOutputProjectionFusions" ||
      name == "FindQkvAttentionOutputProjectionFusions") {
    return kAttentionProjectionDTypeContract;
  }
  if (name == "FindConcatSplitFusions") {
    return kFloatIdentityWithFp32ReduceDTypeContract;
  }
  if (name == "FindConcatSplitFusions" || name == "FindConcatReshapeFusions" ||
      name == "FindShapeReshapeFusions" || name == "FindTileConcatFusions" ||
      name == "FindStridedViewFusions") {
    return kFloatIdentityDTypeContract;
  }
  if (name == "FindModuloGatherFusions" ||
      name == "FindBucketizeGatherFusions" ||
      name == "FindLogBucketizeGatherFusions" ||
      name == "FindMaskedEmbeddingLookupFusions") {
    return kEmbeddingGatherDTypeContract;
  }
  if (name == "FindSparseIdToMaskFusions") {
    return kIntegerMaskDTypeContract;
  }
  if (name == "FindReplaceInvalidIdFusions") {
    return kIntegerDTypeContract;
  }
  if (name == "FindMoEFusions" || name == "FindSplitSequenceMoEFusions" ||
      name == "FindParallelEinsumActivationFusions" ||
      name == "FindTargetIdCountEmbeddingFusions" ||
      name == "FindMathConcatLogFusions" ||
      name == "FindSplitUnsqueezeConcatFusions" ||
      name == "FindSplitConcatFusions" || name == "FindParallelLinearFusions" ||
      name == "FindSliceConcatFusions") {
    return kFp32OnlyDTypeContract;
  }
  return kNoFusionDTypeContract;
}

std::optional<ONNXTensorElementDataType> GetTensorElementType(
    Ort::ConstValueInfo value_info) {
  if (value_info == nullptr) {
    return std::nullopt;
  }
  Ort::ConstTypeInfo type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return std::nullopt;
  }
  return type_info.GetTensorTypeAndShapeInfo().GetElementType();
}

const char* FusionDTypeName(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    default:
      return "other";
  }
}

bool IsFloatingStorageType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

bool IsSupportedStorageType(ONNXTensorElementDataType type,
                            const FusionDTypeContract& contract) {
  if (!IsFloatingStorageType(type)) {
    return true;
  }
  return FloatingTypeAllowed(type, contract.float_policy);
}

bool RequireSameElementType(const std::vector<Ort::ConstValueInfo>& value_infos,
                            ONNXTensorElementDataType& elem_type,
                            std::string* reason) {
  bool have_type = false;
  for (Ort::ConstValueInfo value_info : value_infos) {
    auto current_type = GetTensorElementType(value_info);
    if (!current_type.has_value()) {
      if (reason != nullptr) {
        *reason = "value is missing tensor element type";
      }
      return false;
    }
    if (!have_type) {
      elem_type = *current_type;
      have_type = true;
      continue;
    }
    if (*current_type != elem_type) {
      if (reason != nullptr) {
        *reason = "tensor element types must match";
      }
      return false;
    }
  }
  return have_type;
}

}  // namespace musa_ep
