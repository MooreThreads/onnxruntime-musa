#pragma once

#include <musa_runtime.h>

struct MusaProviderOptions {
  int device_id = 0;
  int has_user_compute_stream = 0;
  musaStream_t user_compute_stream = nullptr;
  int use_ep_level_unified_stream = 0;
  int do_copy_in_default_stream = 1;
};
