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
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
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
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  };
}

inline std::vector<const OrtDataType*> UniqueTensorTypes() {
  return {
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
      GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
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
