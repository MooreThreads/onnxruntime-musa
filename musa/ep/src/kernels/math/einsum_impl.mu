#include "math/einsum_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__global__ void EinsumDiagonalKernel(const void* input,
                                     void* output,
                                     int64_t dim,
                                     int32_t element_size) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < dim; index += total_threads) {
    const int64_t input_index = index * dim + index;
    if (element_size == 4) {
      reinterpret_cast<uint32_t*>(output)[index] =
          reinterpret_cast<const uint32_t*>(input)[input_index];
    } else if (element_size == 8) {
      reinterpret_cast<uint64_t*>(output)[index] =
          reinterpret_cast<const uint64_t*>(input)[input_index];
    } else if (element_size == 2) {
      reinterpret_cast<uint16_t*>(output)[index] =
          reinterpret_cast<const uint16_t*>(input)[input_index];
    } else {
      reinterpret_cast<uint8_t*>(output)[index] =
          reinterpret_cast<const uint8_t*>(input)[input_index];
    }
  }
}

__global__ void EinsumBhijHkKernel(const float* lhs,
                                   const float* rhs,
                                   float* output,
                                   int64_t batch,
                                   int64_t h_dim,
                                   int64_t k_dim,
                                   int64_t i_dim,
                                   int64_t j_dim,
                                   int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    int64_t remaining = index;
    const int64_t j = remaining % j_dim;
    remaining /= j_dim;
    const int64_t i = remaining % i_dim;
    remaining /= i_dim;
    const int64_t k = remaining % k_dim;
    remaining /= k_dim;
    const int64_t b = remaining;

    float acc = 0.0f;
    for (int64_t h = 0; h < h_dim; ++h) {
      const int64_t lhs_index = ((b * h_dim + h) * i_dim + i) * j_dim + j;
      const int64_t rhs_index = h * k_dim + k;
      acc += lhs[lhs_index] * rhs[rhs_index];
    }
    output[index] = acc;
  }
}

}  // namespace

musaError_t LaunchMusaEinsumDiagonalKernel(const void* input,
                                           void* output,
                                           int64_t dim,
                                           int32_t element_size,
                                           musaStream_t stream) {
  if (dim == 0) {
    return musaSuccess;
  }
  EinsumDiagonalKernel<<<BlocksForCount(dim), kThreadsPerBlock, 0, stream>>>(
      input, output, dim, element_size);
  return musaGetLastError();
}

musaError_t LaunchMusaEinsumBhijHkKernel(const float* lhs,
                                         const float* rhs,
                                         float* output,
                                         int64_t batch,
                                         int64_t h_dim,
                                         int64_t k_dim,
                                         int64_t i_dim,
                                         int64_t j_dim,
                                         musaStream_t stream) {
  const int64_t total_elements = batch * k_dim * i_dim * j_dim;
  if (total_elements == 0) {
    return musaSuccess;
  }
  EinsumBhijHkKernel<<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                       stream>>>(lhs, rhs, output, batch, h_dim, k_dim, i_dim,
                                 j_dim, total_elements);
  return musaGetLastError();
}
