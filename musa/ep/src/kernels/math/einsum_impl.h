#pragma once

#include "shared_inc/device_kernel_types.h"

musaError_t LaunchMusaEinsumDiagonalKernel(const void* input, void* output,
                                           int64_t dim, int32_t element_size,
                                           musaStream_t stream);

musaError_t LaunchMusaEinsumBhijHkKernel(const float* lhs, const float* rhs,
                                         float* output, int64_t batch,
                                         int64_t h_dim, int64_t k_dim,
                                         int64_t i_dim, int64_t j_dim,
                                         musaStream_t stream);

musaError_t LaunchMusaEinsumBlhwBjhwBhlKernel(const float* lhs,
                                              const float* rhs, float* output,
                                              int64_t batch, int64_t l_dim,
                                              int64_t h_dim, int64_t w_dim,
                                              int64_t j_dim,
                                              musaStream_t stream);

musaError_t LaunchMusaEinsumIlhwBjhwBhlKernel(const float* lhs,
                                              const float* rhs, float* output,
                                              int64_t batch, int64_t i_dim,
                                              int64_t l_dim, int64_t h_dim,
                                              int64_t w_dim, int64_t j_dim,
                                              musaStream_t stream);

musaError_t LaunchMusaEinsumNikBnkBniKernel(const float* lhs, const float* rhs,
                                            float* output, int64_t batch,
                                            int64_t n_dim, int64_t i_dim,
                                            int64_t k_dim, musaStream_t stream);

musaError_t LaunchMusaEinsumBnkNkdBndKernel(const float* lhs, const float* rhs,
                                            float* output, int64_t batch,
                                            int64_t n_dim, int64_t k_dim,
                                            int64_t d_dim, musaStream_t stream);
