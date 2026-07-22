#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaSegmentMaxBroadcastKernel(const int64_t* segment_ids,
                                                const float* values,
                                                float* output, int64_t count,
                                                musaStream_t stream);
