# common/

Reserved for shared, header-only helpers (status conversion, logging shims, small shape /
stride utilities) that need to be visible from both `kernels/` and the top-level EP code.

Currently empty — the existing helpers live next to their only consumer:

- ABI helpers: [`../plugin_ep_utils.h`](../plugin_ep_utils.h)
- Kernel-registration macros + `KernelDefBuilder` helpers: [`../kernels/utils.h`](../kernels/utils.h)
- MUSA error-string helpers: [`../runtime/musa_runtime.h`](../runtime/musa_runtime.h)

Move a helper here once it is genuinely shared across more than one of those areas.
