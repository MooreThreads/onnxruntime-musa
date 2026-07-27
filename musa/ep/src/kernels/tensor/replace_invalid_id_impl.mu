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

#include "shared_inc/musa_kernel_common.mu.h"
#include "tensor/replace_invalid_id_impl.h"

namespace {

template <typename T>
__global__ void ReplaceInvalidIdKernel(const T* __restrict__ input,
                                       T* __restrict__ output, int64_t count,
                                       T threshold, T replacement) {
  const int64_t thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_id; index < count; index += total_threads) {
    const T id = input[index];
    output[index] = id <= threshold ? replacement : id;
  }
}

template <typename T>
musaError_t LaunchReplaceInvalidIdKernel(const T* input, T* output,
                                         int64_t count, T threshold,
                                         T replacement, musaStream_t stream) {
  if (count == 0) {
    return musaSuccess;
  }
  ReplaceInvalidIdKernel<T>
      <<<BlocksForCount(count), kThreadsPerBlock, 0, stream>>>(
          input, output, count, threshold, replacement);
  return musaGetLastError();
}

}  // namespace

musaError_t LaunchMusaReplaceInvalidIdInt32Kernel(
    const int32_t* input, int32_t* output, int64_t count, int32_t threshold,
    int32_t replacement, musaStream_t stream) {
  return LaunchReplaceInvalidIdKernel(input, output, count, threshold,
                                      replacement, stream);
}

musaError_t LaunchMusaReplaceInvalidIdInt64Kernel(
    const int64_t* input, int64_t* output, int64_t count, int64_t threshold,
    int64_t replacement, musaStream_t stream) {
  return LaunchReplaceInvalidIdKernel(input, output, count, threshold,
                                      replacement, stream);
}
