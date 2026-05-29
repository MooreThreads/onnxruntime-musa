#pragma once

#include <musa_runtime.h>

#include <string>

inline const char* MusaErrorString(musaError_t status) {
  const char* msg = musaGetErrorString(status);
  return msg != nullptr ? msg : "unknown MUSA error";
}
