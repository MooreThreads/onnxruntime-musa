#pragma once

#include <cstdint>

#include <musa_runtime.h>

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
};

enum class MusaCompareOp : int32_t {
  Equal = 0,
  Greater = 1,
};

enum class MusaReduceOp : int32_t {
  Prod = 0,
  Sum = 1,
  Mean = 2,
  SumSquare = 3,
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
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_strides[kMusaMaxBroadcastRank];
};

struct MusaTransposeParams {
  int32_t rank;
  int64_t total_elements;
  int64_t input_strides[kMusaMaxBroadcastRank];
  int64_t output_dims[kMusaMaxBroadcastRank];
  int32_t perm[kMusaMaxBroadcastRank];
};

musaError_t LaunchMusaBinaryFloatKernel(const float* lhs,
                                        const float* rhs,
                                        float* output,
                                        MusaBroadcastParams params,
                                        MusaBinaryOp op,
                                        musaStream_t stream);

musaError_t LaunchMusaUnaryFloatKernel(const float* input,
                                       float* output,
                                       int64_t count,
                                       MusaUnaryOp op,
                                       float alpha,
                                       musaStream_t stream);

musaError_t LaunchMusaCompareFloatKernel(const float* lhs,
                                         const float* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream);

musaError_t LaunchMusaCompareInt32Kernel(const int32_t* lhs,
                                         const int32_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream);

musaError_t LaunchMusaCompareInt64Kernel(const int64_t* lhs,
                                         const int64_t* rhs,
                                         uint8_t* output,
                                         MusaBroadcastParams params,
                                         MusaCompareOp op,
                                         musaStream_t stream);

musaError_t LaunchMusaNotBoolKernel(const uint8_t* input,
                                    uint8_t* output,
                                    int64_t count,
                                    musaStream_t stream);

musaError_t LaunchMusaOrBoolKernel(const uint8_t* lhs,
                                   const uint8_t* rhs,
                                   uint8_t* output,
                                   MusaBroadcastParams params,
                                   musaStream_t stream);

musaError_t LaunchMusaCastInt32ToFloatKernel(const int32_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream);

musaError_t LaunchMusaCastInt64ToFloatKernel(const int64_t* input,
                                             float* output,
                                             int64_t count,
                                             musaStream_t stream);

musaError_t LaunchMusaSliceKernel(const void* input,
                                  void* output,
                                  int32_t element_size,
                                  MusaSliceParams params,
                                  musaStream_t stream);

musaError_t LaunchMusaGatherKernel(const void* input,
                                   const void* indices,
                                   void* output,
                                   int32_t element_size,
                                   int32_t index_element_size,
                                   int64_t input_block_size,
                                   int64_t indices_max,
                                   int64_t output_block_size,
                                   int64_t block_size,
                                   int64_t output_count,
                                   musaStream_t stream);

musaError_t LaunchMusaReduceFloatKernel(const float* input,
                                        float* output,
                                        MusaReduceParams params,
                                        MusaReduceOp op,
                                        musaStream_t stream);

musaError_t LaunchMusaBatchNormalizationFloatKernel(const float* input,
                                                    const float* scale,
                                                    const float* bias,
                                                    const float* mean,
                                                    const float* variance,
                                                    float* output,
                                                    MusaBatchNormParams params,
                                                    musaStream_t stream);

musaError_t LaunchMusaTransposeKernel(const void* input,
                                      void* output,
                                      int32_t element_size,
                                      MusaTransposeParams params,
                                      musaStream_t stream);

musaError_t LaunchMusaSoftmaxFloatKernel(const float* input,
                                         float* output,
                                         int64_t outer,
                                         int64_t dim,
                                         int64_t inner,
                                         musaStream_t stream);
