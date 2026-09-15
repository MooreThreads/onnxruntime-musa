#pragma once

// MUSA 5.1 ships the C++ API in a separate mudnncxx include directory,
// while 4.3.8 exposes the same API through mudnn.h.
#if __has_include(<mudnncxx/mudnn.h>)
#include <mudnncxx/mudnn.h>
#else
#include <mudnn.h>
#endif
