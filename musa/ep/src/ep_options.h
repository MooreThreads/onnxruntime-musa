#pragma once

#include <musa_runtime.h>

#include <string>

enum class MusaPrecisionPolicy {
  Strict,
  Report,
};

struct MusaProviderOptions {
  int device_id = 0;
  int has_user_compute_stream = 0;
  musaStream_t user_compute_stream = nullptr;
  int use_ep_level_unified_stream = 0;
  int do_copy_in_default_stream = 1;
  MusaPrecisionPolicy precision_policy = MusaPrecisionPolicy::Strict;
  int dtype_diagnostics = 0;
};

inline const char* MusaPrecisionPolicyName(MusaPrecisionPolicy policy) {
  return policy == MusaPrecisionPolicy::Report ? "report" : "strict";
}
