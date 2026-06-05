#include "math/unary_elementwise_ops_impl.h"
#include "shared_inc/musa_kernel_common.mu.h"

namespace {

__device__ __forceinline__ float UnaryValue(float x, MusaUnaryOp op, float alpha) {
  switch (op) {
    case MusaUnaryOp::Relu:
      return x > 0.0f ? x : 0.0f;
    case MusaUnaryOp::LeakyRelu:
      return x >= 0.0f ? x : alpha * x;
    case MusaUnaryOp::Sqrt:
      return sqrtf(x);
    case MusaUnaryOp::Reciprocal:
      return 1.0f / x;
    case MusaUnaryOp::Neg:
      return -x;
    case MusaUnaryOp::Log:
      return logf(x);
    case MusaUnaryOp::Tanh:
      return tanhf(x);
    case MusaUnaryOp::Sigmoid:
      return 1.0f / (1.0f + expf(-x));
    case MusaUnaryOp::Abs:
      return fabsf(x);
    case MusaUnaryOp::Erf:
      return erff(x);
    case MusaUnaryOp::Exp:
      return expf(x);
    case MusaUnaryOp::Sign:
      return (x > 0.0f) - (x < 0.0f);
    case MusaUnaryOp::IsNaN:
      return isnan(x) ? 1.0f : 0.0f;
    case MusaUnaryOp::Round:
      return nearbyintf(x);
    case MusaUnaryOp::Softplus:
      return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
    case MusaUnaryOp::Ceil:
      return ceilf(x);
  }
  return x;
}

template <typename T>
__device__ __forceinline__ T UnaryValueTyped(T x, MusaUnaryOp op, float alpha) {
  switch (op) {
    case MusaUnaryOp::Relu:
      return x > static_cast<T>(0) ? x : static_cast<T>(0);
    case MusaUnaryOp::LeakyRelu:
      return x >= static_cast<T>(0)
                 ? x
                 : static_cast<T>(static_cast<double>(alpha) *
                                  static_cast<double>(x));
    case MusaUnaryOp::Sqrt:
      return static_cast<T>(sqrt(static_cast<double>(x)));
    case MusaUnaryOp::Reciprocal:
      return static_cast<T>(1.0 / static_cast<double>(x));
    case MusaUnaryOp::Neg:
      return static_cast<T>(-x);
    case MusaUnaryOp::Log:
      return static_cast<T>(log(static_cast<double>(x)));
    case MusaUnaryOp::Tanh:
      return static_cast<T>(tanh(static_cast<double>(x)));
    case MusaUnaryOp::Sigmoid:
      return static_cast<T>(1.0 / (1.0 + exp(-static_cast<double>(x))));
    case MusaUnaryOp::Abs:
      return x < static_cast<T>(0) ? static_cast<T>(-x) : x;
    case MusaUnaryOp::Erf:
      return static_cast<T>(erf(static_cast<double>(x)));
    case MusaUnaryOp::Exp:
      return static_cast<T>(exp(static_cast<double>(x)));
    case MusaUnaryOp::Sign:
      return static_cast<T>((x > static_cast<T>(0)) -
                            (x < static_cast<T>(0)));
    case MusaUnaryOp::IsNaN:
      return static_cast<T>(isnan(static_cast<double>(x)) ? 1 : 0);
    case MusaUnaryOp::Round:
      return static_cast<T>(nearbyint(static_cast<double>(x)));
    case MusaUnaryOp::Softplus: {
      const double value = static_cast<double>(x);
      return static_cast<T>(log1p(exp(-fabs(value))) + fmax(value, 0.0));
    }
    case MusaUnaryOp::Ceil:
      return static_cast<T>(ceil(static_cast<double>(x)));
  }
  return x;
}

template <typename T>
__global__ void UnaryKernel(const T* input,
                            T* output,
                            int64_t count,
                            MusaUnaryOp op,
                            float alpha) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = UnaryValueTyped(input[index], op, alpha);
  }
}

__global__ void UnaryFloatKernel(const float* input,
                                 float* output,
                                 int64_t count,
                                 MusaUnaryOp op,
                                 float alpha) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = UnaryValue(input[index], op, alpha);
  }
}

template <typename T>
__global__ void UnaryFloatLikeKernel(const T* input,
                                     T* output,
                                     int64_t count,
                                     MusaUnaryOp op,
                                     float alpha) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    const float value = MusaScalarToFloat(input[index]);
    output[index] = MusaScalarFromFloat<T>(UnaryValue(value, op, alpha));
  }
}

template <typename T>
__global__ void IsNaNKernel(const T* input,
                            uint8_t* output,
                            int64_t count) {
  const int64_t thread_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    output[index] = static_cast<uint8_t>(isnan(MusaScalarToDouble(input[index])));
  }
}

}  // namespace

template <typename T>
musaError_t LaunchUnaryTyped(const void* input,
                             void* output,
                             int64_t count,
                             MusaUnaryOp op,
                             float alpha,
                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  UnaryKernel<T><<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count,
      op, alpha);
  return musaGetLastError();
}

template <typename T>
musaError_t LaunchUnaryFloatLikeTyped(const void* input,
                                      void* output,
                                      int64_t count,
                                      MusaUnaryOp op,
                                      float alpha,
                                      musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  UnaryFloatLikeKernel<T><<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), count,
      op, alpha);
  return musaGetLastError();
}

musaError_t LaunchMusaUnaryKernel(const void* input,
                                  void* output,
                                  int64_t count,
                                  MusaUnaryOp op,
                                  float alpha,
                                  MusaElementType elem_type,
                                  musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchUnaryTyped<float>(input, output, count, op, alpha, stream);
    case MusaElementType::Double:
      return LaunchUnaryTyped<double>(input, output, count, op, alpha, stream);
    case MusaElementType::Uint8:
      return LaunchUnaryTyped<uint8_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Uint16:
      return LaunchUnaryTyped<uint16_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Uint32:
      return LaunchUnaryTyped<uint32_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Uint64:
      return LaunchUnaryTyped<uint64_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Int8:
      return LaunchUnaryTyped<int8_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Int16:
      return LaunchUnaryTyped<int16_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Int32:
      return LaunchUnaryTyped<int32_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Int64:
      return LaunchUnaryTyped<int64_t>(input, output, count, op, alpha, stream);
    case MusaElementType::Float16:
      return LaunchUnaryFloatLikeTyped<__half>(input, output, count, op, alpha,
                                               stream);
    case MusaElementType::BFloat16:
      return LaunchUnaryFloatLikeTyped<__mt_bfloat16>(
          input, output, count, op, alpha, stream);
    default:
      return musaErrorNotSupported;
  }
}

musaError_t LaunchMusaUnaryFloatKernel(const float* input,
                                       float* output,
                                       int64_t count,
                                       MusaUnaryOp op,
                                       float alpha,
                                       musaStream_t stream) {
  return LaunchMusaUnaryKernel(input, output, count, op, alpha,
                               MusaElementType::Float, stream);
}

template <typename T>
musaError_t LaunchIsNaNTyped(const void* input,
                             uint8_t* output,
                             int64_t count,
                             musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  IsNaNKernel<T><<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input), output, count);
  return musaGetLastError();
}

musaError_t LaunchMusaIsNaNKernel(const void* input,
                                  uint8_t* output,
                                  int64_t count,
                                  MusaElementType elem_type,
                                  musaStream_t stream) {
  switch (elem_type) {
    case MusaElementType::Float:
      return LaunchIsNaNTyped<float>(input, output, count, stream);
    case MusaElementType::Double:
      return LaunchIsNaNTyped<double>(input, output, count, stream);
    case MusaElementType::Float16:
      return LaunchIsNaNTyped<__half>(input, output, count, stream);
    case MusaElementType::BFloat16:
      return LaunchIsNaNTyped<__mt_bfloat16>(input, output, count, stream);
    default:
      return musaErrorNotSupported;
  }
}
