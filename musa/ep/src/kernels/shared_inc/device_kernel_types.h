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
  Ceil = 15,
  Floor = 16,
};

enum class MusaCompareOp : int32_t {
  Equal = 0,
  Greater = 1,
  Less = 2,
  GreaterOrEqual = 3,
  LessOrEqual = 4,
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
  L2 = 5,
};

struct MusaBatchNormParams {
  int64_t total_elements;
  int64_t channels;
  int64_t spatial_size;
  float epsilon;
};

struct MusaGlobalAveragePoolParams {
  int64_t output_elements;
  int64_t channels;
  int64_t spatial_elements;
};

struct MusaMaxPoolParams {
  int32_t rank;
  int32_t spatial_rank;
  int32_t has_indices;
  int64_t output_elements;
  int64_t input_dims[kMusaMaxBroadcastRank];
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_dims[kMusaMaxBroadcastRank];
  int64_t output_strides[kMusaMaxBroadcastRank];
  int64_t kernel_shape[kMusaMaxBroadcastRank];
  int64_t pads_begin[kMusaMaxBroadcastRank];
  int64_t strides[kMusaMaxBroadcastRank];
  int64_t dilations[kMusaMaxBroadcastRank];
};

struct MusaRangeParams {
  int64_t count;
};

struct MusaClipParams {
  int64_t count;
  const void* min_data;
  const void* max_data;
  int32_t has_min;
  int32_t has_max;
};

struct MusaLayerNormParams {
  int64_t rows;
  int64_t norm_size;
  int32_t has_bias;
};

struct MusaAttentionParams {
  int64_t batch_size;
  int64_t sequence_length;
  int64_t input_hidden_size;
  int64_t q_hidden_size;
  int64_t k_hidden_size;
  int64_t v_hidden_size;
  int64_t q_head_size;
  int64_t k_head_size;
  int64_t v_head_size;
  int64_t num_heads;
  int64_t qkv_hidden_size;
  int64_t mask_batch;
  int64_t mask_heads;
  float scale;
  int32_t has_mask;
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
  int32_t reduce_axes_count;
  int64_t output_elements;
  int64_t reduce_dim;
  int64_t reduction_elements;
  int64_t inner_size;
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

struct MusaGatherNDParams {
  int32_t input_rank;
  int32_t indices_rank;
  int32_t batch_dims;
  int32_t num_slice_dims;
  int64_t output_elements;
  int64_t slice_size;
  int64_t num_slices_per_batch;
  int64_t input_batch_stride;
  int64_t input_dims[kMusaMaxBroadcastRank];
  int64_t sizes_from_slice_dims[kMusaMaxBroadcastRank];
};

struct MusaGatherElementsParams {
  int32_t rank;
  int32_t axis;
  int64_t output_elements;
  int64_t data_dims[kMusaMaxBroadcastRank];
  int64_t data_strides[kMusaMaxBroadcastRank];
  int64_t indices_strides[kMusaMaxBroadcastRank];
};

struct MusaScatterNDParams {
  int32_t last_index_dimension;
  int32_t reduction;
  int64_t num_indices;
  int64_t updates_slice_size;
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t input_dims[kMusaMaxBroadcastRank];
};

struct MusaScatterElementsParams {
  int32_t rank;
  int32_t axis;
  int32_t reduction;
  int64_t updates_elements;
  int64_t data_dims[kMusaMaxBroadcastRank];
  int64_t data_strides[kMusaMaxBroadcastRank];
  int64_t updates_strides[kMusaMaxBroadcastRank];
};

struct MusaReverseSequenceParams {
  int64_t batch_size;
  int64_t max_seq_len;
  int64_t element_size;
  int64_t total_elements;
  int32_t time_major;
};

struct MusaTopKParams {
  int64_t rows;
  int64_t dim;
  int64_t inner;
  int64_t k;
  int64_t output_elements;
  int32_t largest;
  int32_t sorted;
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
