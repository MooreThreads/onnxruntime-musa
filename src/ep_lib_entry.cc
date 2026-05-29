#include <memory>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "ep_factory.h"

#if defined(_WIN32)
#define ORT_MUSA_EXPORT __declspec(dllexport)
#else
#define ORT_MUSA_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

ORT_MUSA_EXPORT OrtStatus* CreateEpFactories(const char*,
                                             const OrtApiBase* ort_api_base,
                                             const OrtLogger* default_logger,
                                             OrtEpFactory** factories,
                                             size_t max_factories,
                                             size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    return ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MusaExecutionProvider requires space for one factory.");
  }

  auto factory =
      std::make_unique<MusaEpFactory>(*ort_api, *ep_api, *default_logger);
  factories[0] = factory.release();
  *num_factories = 1;
  return nullptr;
}

ORT_MUSA_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<MusaEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
