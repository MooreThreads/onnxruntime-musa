#include "supported_ops.h"

bool MusaSupportsOp(const char* op_type) {
  return op_type != nullptr && std::string(op_type) == "Relu";
}
