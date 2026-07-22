#pragma once

#include <cstddef>

#include "shared_inc/device_kernel_types.h"

constexpr int64_t kMusaTopKBlockSortMaxDim = 1024;
constexpr int64_t kMusaTopKStablePostprocessMaxK = 1024;

musaError_t LaunchMusaTopKPairReduceKernel(const void* input, void* values,
                                           int64_t* indices,
                                           MusaTopKParams params,
                                           MusaElementType elem_type,
                                           musaStream_t stream);

musaError_t LaunchMusaTopKBlockSortKernel(const void* input, void* values,
                                          int64_t* indices,
                                          MusaTopKParams params,
                                          MusaElementType elem_type,
                                          musaStream_t stream);

musaError_t LaunchMusaTopKStablePostprocessKernel(
    const void* input, void* values, int64_t* indices, MusaTopKParams params,
    MusaElementType elem_type, musaStream_t stream);

musaError_t GetMusaTopKRadixSortWorkspaceSize(MusaTopKParams params,
                                              MusaElementType elem_type,
                                              size_t* workspace_bytes);

musaError_t LaunchMusaTopKRadixSortKernel(
    const void* input, void* values, int64_t* indices, MusaTopKParams params,
    MusaElementType elem_type, void* workspace, size_t workspace_bytes,
    musaStream_t stream);
