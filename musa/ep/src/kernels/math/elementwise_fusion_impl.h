#pragma once

#include <cstdint>

#include <musa_runtime.h>

constexpr int32_t kMusaElementwiseFusionMaxOps = 768;
constexpr int32_t kMusaElementwiseFusionMaxInputs = 2048;
constexpr int32_t kMusaElementwiseFusionMaxOutputs = 256;

enum class MusaElementwiseFusionOp : int32_t {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
  Pow = 4,
  Max = 5,
  Min = 6,
  Exp = 7,
  Log = 8,
  Sigmoid = 9,
  Tanh = 10,
  Sqrt = 11,
  Reciprocal = 12,
  Neg = 13,
  Where = 14,
  IsNaN = 15,
};

enum class MusaElementwiseFusionOperandKind : int32_t {
  FloatInput = 0,
  FloatScalar = 1,
  FloatTemp = 2,
  BoolInput = 3,
  BoolScalar = 4,
  BoolTemp = 5,
};

struct MusaElementwiseFusionOperand {
  int32_t kind;
  int32_t index;
  float scalar;
  uint8_t bool_scalar;
};

struct MusaElementwiseFusionNode {
  int32_t op;
  int32_t input_count;
  MusaElementwiseFusionOperand inputs[3];
  uint8_t output_is_bool;
};

musaError_t LaunchMusaElementwiseFusionKernel(
    const MusaElementwiseFusionNode* ops, int32_t op_count,
    const void* const* input_ptrs, const uint8_t* input_is_scalar,
    void* const* output_ptrs, const int32_t* output_refs,
    const uint8_t* output_is_bool, int32_t output_count, int64_t rows,
    musaStream_t stream);
