#pragma once

#include <musa_runtime.h>

#include <cstdint>

constexpr int kMusaMaxBroadcastRank = 8;

enum class MusaBinaryOp : int32_t {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
  Pow = 4,
  Max = 5,
  Min = 6,
};

enum class MusaUnaryOp : int32_t {
  Relu = 0,
  LeakyRelu = 1,
  Sqrt = 2,
  Reciprocal = 3,
  Neg = 4,
  Log = 5,
  Tanh = 6,
  Sigmoid = 7,
  Abs = 8,
  Erf = 9,
  Exp = 10,
  Sign = 11,
  IsNaN = 12,
  Round = 13,
  Softplus = 14,
};

enum class MusaCompareOp : int32_t {
  Equal = 0,
  Greater = 1,
  Less = 2,
  GreaterOrEqual = 3,
};

enum class MusaElementType : int32_t {
  Float = 1,
  Uint8 = 2,
  Int8 = 3,
  Uint16 = 4,
  Int16 = 5,
  Int32 = 6,
  Int64 = 7,
  String = 8,
  Bool = 9,
  Float16 = 10,
  Double = 11,
  Uint32 = 12,
  Uint64 = 13,
  BFloat16 = 16,
};

enum class MusaReduceOp : int32_t {
  Prod = 0,
  Sum = 1,
  Mean = 2,
  SumSquare = 3,
  Max = 4,
};

struct MusaBatchNormParams {
  int64_t total_elements;
  int64_t channels;
  int64_t spatial_size;
  float epsilon;
};

struct MusaBroadcastParams {
  int32_t rank;
  int64_t total_elements;
  int64_t output_strides[kMusaMaxBroadcastRank];
  int64_t lhs_strides[kMusaMaxBroadcastRank];
  int64_t rhs_strides[kMusaMaxBroadcastRank];
};

struct MusaSliceParams {
  int32_t rank;
  int64_t total_elements;
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_dims[kMusaMaxBroadcastRank];
  int64_t starts[kMusaMaxBroadcastRank];
  int64_t steps[kMusaMaxBroadcastRank];
};

struct MusaReduceParams {
  int32_t rank;
  int32_t reduce_axis;
  int64_t output_elements;
  int64_t reduce_dim;
  int64_t reduction_elements;
  int64_t input_dims[kMusaMaxBroadcastRank];
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_strides[kMusaMaxBroadcastRank];
  int32_t reduce_axes[kMusaMaxBroadcastRank];
};

struct MusaTransposeParams {
  int32_t rank;
  int64_t total_elements;
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_dims[kMusaMaxBroadcastRank];
  int32_t perm[kMusaMaxBroadcastRank];
};

struct MusaTileParams {
  int32_t rank;
  int64_t total_elements;
  int64_t input_dims[kMusaMaxBroadcastRank];
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_dims[kMusaMaxBroadcastRank];
};

struct MusaWhereParams {
  int32_t rank;
  int64_t total_elements;
  int64_t output_strides[kMusaMaxBroadcastRank];
  int64_t condition_strides[kMusaMaxBroadcastRank];
  int64_t x_strides[kMusaMaxBroadcastRank];
  int64_t y_strides[kMusaMaxBroadcastRank];
};

struct MusaPadParams {
  int32_t rank;
  int64_t total_elements;
  int64_t input_dims[kMusaMaxBroadcastRank];
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_dims[kMusaMaxBroadcastRank];
  int64_t pads_begin[kMusaMaxBroadcastRank];
};

struct MusaNonZeroParams {
  int32_t rank;
  int64_t total_elements;
  int64_t nonzero_elements;
  int64_t input_strides[kMusaMaxBroadcastRank];
};

struct MusaConv2DParams {
  int64_t n;
  int64_t c;
  int64_t h;
  int64_t w;
  int64_t m;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t out_h;
  int64_t out_w;
  int64_t pad_h;
  int64_t pad_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t total_elements;
};
