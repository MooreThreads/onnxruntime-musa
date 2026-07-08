// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <musa_runtime.h>

#include <cstdint>

musaError_t LaunchMusaOneHotKernel(const void* indices, void* output,
                                   int32_t indices_type, int32_t element_size,
                                   uint64_t off_value, uint64_t on_value,
                                   int64_t depth, int64_t suffix,
                                   int64_t total_elements, musaStream_t stream);
