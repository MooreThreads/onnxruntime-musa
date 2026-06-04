#include "math/variadic_elementwise_ops_impl.h"

#include "math/binary_elementwise_ops_impl.h"

musaError_t LaunchMusaVariadicSumKernel(const void* lhs,
                                        const void* rhs,
                                        void* output,
                                        MusaBroadcastParams params,
                                        MusaElementType elem_type,
                                        musaStream_t stream) {
  return LaunchMusaBinaryKernel(lhs, rhs, output, params, MusaBinaryOp::Add,
                                elem_type, stream);
}
