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

__global__ void EinsumBlhwBjhwBhlKernel(const float* lhs,
                                        const float* rhs,
                                        float* output,
                                        int64_t batch,
                                        int64_t l_dim,
                                        int64_t h_dim,
                                        int64_t w_dim,
                                        int64_t j_dim,
                                        int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    int64_t remaining = index;
    const int64_t l = remaining % l_dim;
    remaining /= l_dim;
    const int64_t h = remaining % h_dim;
    remaining /= h_dim;
    const int64_t b = remaining;

    float acc = 0.0f;
    for (int64_t j = 0; j < j_dim; ++j) {
      for (int64_t w = 0; w < w_dim; ++w) {
        const int64_t lhs_index = ((b * l_dim + l) * h_dim + h) * w_dim + w;
        const int64_t rhs_index = ((b * j_dim + j) * h_dim + h) * w_dim + w;
        acc += lhs[lhs_index] * rhs[rhs_index];
      }
    }
    output[index] = acc;
  }
}

__global__ void EinsumIlhwBjhwBhlKernel(const float* lhs,
                                        const float* rhs,
                                        float* output,
                                        int64_t batch,
                                        int64_t i_dim,
                                        int64_t l_dim,
                                        int64_t h_dim,
                                        int64_t w_dim,
                                        int64_t j_dim,
                                        int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    int64_t remaining = index;
    const int64_t l = remaining % l_dim;
    remaining /= l_dim;
    const int64_t h = remaining % h_dim;
    remaining /= h_dim;
    const int64_t b = remaining;

    float acc = 0.0f;
    for (int64_t i = 0; i < i_dim; ++i) {
      for (int64_t j = 0; j < j_dim; ++j) {
        for (int64_t w = 0; w < w_dim; ++w) {
          const int64_t lhs_index =
              ((i * l_dim + l) * h_dim + h) * w_dim + w;
          const int64_t rhs_index =
              ((b * j_dim + j) * h_dim + h) * w_dim + w;
          acc += lhs[lhs_index] * rhs[rhs_index];
        }
      }
    }
    output[index] = acc;
  }
}

__global__ void EinsumNikBnkBniKernel(const float* lhs,
                                      const float* rhs,
                                      float* output,
                                      int64_t batch,
                                      int64_t n_dim,
                                      int64_t i_dim,
                                      int64_t k_dim,
                                      int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    int64_t remaining = index;
    const int64_t i = remaining % i_dim;
    remaining /= i_dim;
    const int64_t n = remaining % n_dim;
    remaining /= n_dim;
    const int64_t b = remaining;

    float acc = 0.0f;
    for (int64_t k = 0; k < k_dim; ++k) {
      const int64_t lhs_index = (n * i_dim + i) * k_dim + k;
      const int64_t rhs_index = (b * n_dim + n) * k_dim + k;
      acc += lhs[lhs_index] * rhs[rhs_index];
    }
    output[index] = acc;
  }
}

__global__ void EinsumBnkNkdBndKernel(const float* lhs,
                                      const float* rhs,
                                      float* output,
                                      int64_t batch,
                                      int64_t n_dim,
                                      int64_t k_dim,
                                      int64_t d_dim,
                                      int64_t total_elements) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < total_elements;
       index += total_threads) {
    int64_t remaining = index;
    const int64_t d = remaining % d_dim;
    remaining /= d_dim;
    const int64_t n = remaining % n_dim;
    remaining /= n_dim;
    const int64_t b = remaining;

    float acc = 0.0f;
    for (int64_t k = 0; k < k_dim; ++k) {
      const int64_t lhs_index = (b * n_dim + n) * k_dim + k;
      const int64_t rhs_index = (n * k_dim + k) * d_dim + d;
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

musaError_t LaunchMusaEinsumBlhwBjhwBhlKernel(const float* lhs,
                                              const float* rhs,
                                              float* output,
                                              int64_t batch,
                                              int64_t l_dim,
                                              int64_t h_dim,
                                              int64_t w_dim,
                                              int64_t j_dim,
                                              musaStream_t stream) {
  const int64_t total_elements = batch * h_dim * l_dim;
  if (total_elements == 0) {
    return musaSuccess;
  }
  EinsumBlhwBjhwBhlKernel<<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                            stream>>>(lhs, rhs, output, batch, l_dim, h_dim,
                                      w_dim, j_dim, total_elements);
  return musaGetLastError();
}

musaError_t LaunchMusaEinsumIlhwBjhwBhlKernel(const float* lhs,
                                              const float* rhs,
                                              float* output,
                                              int64_t batch,
                                              int64_t i_dim,
                                              int64_t l_dim,
                                              int64_t h_dim,
                                              int64_t w_dim,
                                              int64_t j_dim,
                                              musaStream_t stream) {
  const int64_t total_elements = batch * h_dim * l_dim;
  if (total_elements == 0) {
    return musaSuccess;
  }
  EinsumIlhwBjhwBhlKernel<<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                            stream>>>(lhs, rhs, output, batch, i_dim, l_dim,
                                      h_dim, w_dim, j_dim, total_elements);
  return musaGetLastError();
}

musaError_t LaunchMusaEinsumNikBnkBniKernel(const float* lhs,
                                             const float* rhs,
                                             float* output,
                                             int64_t batch,
                                             int64_t n_dim,
                                             int64_t i_dim,
                                             int64_t k_dim,
                                             musaStream_t stream) {
  const int64_t total_elements = batch * n_dim * i_dim;
  if (total_elements == 0) {
    return musaSuccess;
  }
  EinsumNikBnkBniKernel<<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                          stream>>>(lhs, rhs, output, batch, n_dim, i_dim,
                                    k_dim, total_elements);
  return musaGetLastError();
}

musaError_t LaunchMusaEinsumBnkNkdBndKernel(const float* lhs,
                                             const float* rhs,
                                             float* output,
                                             int64_t batch,
                                             int64_t n_dim,
                                             int64_t k_dim,
                                             int64_t d_dim,
                                             musaStream_t stream) {
  const int64_t total_elements = batch * n_dim * d_dim;
  if (total_elements == 0) {
    return musaSuccess;
  }
  EinsumBnkNkdBndKernel<<<BlocksForCount(total_elements), kThreadsPerBlock, 0,
                          stream>>>(lhs, rhs, output, batch, n_dim, k_dim,
                                    d_dim, total_elements);
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
