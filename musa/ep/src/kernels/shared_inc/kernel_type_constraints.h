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

#include <vector>

#include "utils.h"

// Type-constraint helpers. The names are also parsed by
// scripts/gen_supported_ops.py to render musa/docs/supported_ops.md.
inline std::vector<const OrtDataType*> AllTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

// Everything AllTensorTypes() covers except bfloat16, i.e. the type set of the
// ONNX schemas that predate opset 13.
inline std::vector<const OrtDataType*> AllTensorTypesNoBFloat16() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> TensorTypesWithBool() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> AllFixedSizeTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> AllFixedSizeTensorTypesNoBFloat16() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

// ONNX binary elementwise ops through opset 12 share the opset-13 type set
// without bfloat16, which the schemas only allow from opset 13 on.
inline std::vector<const OrtDataType*> BinaryNumericOpset7TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> BinaryNumericOpset13TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> BinaryNumericOpset14TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

// Pow opset 12: integer bases and exponents join the floating-point types, and
// the exponent may differ from the base in type. bfloat16 only arrives with the
// opset-15 schema.
inline std::vector<const OrtDataType*> PowOpset12TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> PowTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> PowExponentOpset13TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> VariadicMinMaxTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> FloatLikeTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

// muDNN MatMul/BatchMatMul device dispatch excludes DOUBLE.
inline std::vector<const OrtDataType*> MatMulTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

// ONNX random generator schemas through opset 19 do not include BF16.
inline std::vector<const OrtDataType*> RandomTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> FloatTensorTypes() {
  return {GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)};
}

inline std::vector<const OrtDataType*> ClipTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

// The opset-11/12 Clip and Pad schemas: every fixed-size numeric type but bool
// (which Pad only gains at opset 13) and bfloat16 (opset 13).
inline std::vector<const OrtDataType*>
AllFixedSizeTensorTypesNoBoolNoBFloat16() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> PReluOpset16TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> RangeTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

// ReduceMean / ReduceL2 through opset 12: the floating-point subset plus
// int32 that the pre-opset-13 schemas allow and the kernels implement (the
// schemas only gain bfloat16 at opset 13).
inline std::vector<const OrtDataType*> ReduceFloatOpset1TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
  };
}

inline std::vector<const OrtDataType*> ReduceMeanTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
  };
}

inline std::vector<const OrtDataType*> ReduceL2Opset13TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
  };
}

inline std::vector<const OrtDataType*> HfdTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> MaxPoolOpset12TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
  };
}

inline std::vector<const OrtDataType*> TopKTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> UniqueTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> GatherNDTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

// Abs opset 6-12: same as the opset-13 set without bfloat16.
inline std::vector<const OrtDataType*> AbsOpset6TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> AbsTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

// Neg opset 6-12: same as the opset-13 set without bfloat16.
inline std::vector<const OrtDataType*> NegOpset6TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> NegTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> SignTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> BitwiseIntegerTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

// ReduceMax through opset 12: the opset-13 set without bfloat16.
inline std::vector<const OrtDataType*> ReduceMaxOpset1TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> ReduceMaxTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

// Less / Greater opset 7-8 only accept floating-point tensors; integer support
// arrives with the opset-9 schema.
inline std::vector<const OrtDataType*> CompareOpset7TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> CompareTensorTypesNoBFloat16() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  };
}

inline std::vector<const OrtDataType*> CompareTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

// Equal opset 7-10 accepts bool and the integer types only; the remaining
// numeric types arrive with the opset-11 schema.
inline std::vector<const OrtDataType*> EqualOpset7TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

// Equal opset 11-12: the opset-13 set without bfloat16.
inline std::vector<const OrtDataType*> EqualOpset11TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> EqualTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  };
}

inline std::vector<const OrtDataType*> FloatBoolTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
  };
}

inline std::vector<const OrtDataType*> IntTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> BoolTensorTypes() {
  return {GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)};
}

inline std::vector<const OrtDataType*> StringTensorTypes() {
  return {GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)};
}

inline std::vector<const OrtDataType*> NonZeroTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
  };
}

inline std::vector<const OrtDataType*> WhereOpset9TensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
  };
}
