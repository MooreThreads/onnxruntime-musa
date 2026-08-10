// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <exception>
#include <memory>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "ep_factory.h"

namespace {

OrtStatus* CreateBootstrapStatus(const OrtApiBase* ort_api_base,
                                 OrtErrorCode code,
                                 const char* message) noexcept {
  if (ort_api_base == nullptr) {
    return nullptr;
  }

  // CreateStatus is the first OrtApi entry and is available from API version 1.
  // Use that baseline API to report that the host is too old for the required
  // API.
  const OrtApi* status_api = ort_api_base->GetApi(1);
  return status_api != nullptr ? status_api->CreateStatus(code, message)
                               : nullptr;
}

}  // namespace

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* /*registration_name*/,
                                           const OrtApiBase* ort_api_base,
                                           const OrtLogger* default_logger,
                                           OrtEpFactory** factories,
                                           size_t max_factories,
                                           size_t* num_factories) {
  if (num_factories != nullptr) {
    *num_factories = 0;
  }

  // A null OrtApiBase violates the Plugin EP entry-point contract. There is no
  // OrtApi available to allocate an OrtStatus in that case, so fail closed by
  // advertising no factories.
  if (ort_api_base == nullptr) {
    return nullptr;
  }

  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  if (ort_api == nullptr) {
    const char* host_version = ort_api_base->GetVersionString();
    char message[256];
    std::snprintf(message, sizeof(message),
                  "MUSAExecutionProvider requires ORT API version %u; host "
                  "runtime version is %s and does not provide that API",
                  static_cast<unsigned>(ORT_API_VERSION),
                  host_version != nullptr ? host_version : "unknown");
    return CreateBootstrapStatus(ort_api_base, ORT_FAIL, message);
  }

  if (factories == nullptr || num_factories == nullptr ||
      default_logger == nullptr) {
    return ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MUSAExecutionProvider received null Plugin EP output/logger pointers");
  }

  if (max_factories < 1) {
    return ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Not enough space to return EP factory. Need at least one.");
  }

  const OrtEpApi* ep_api = ort_api->GetEpApi();
  if (ep_api == nullptr) {
    return ort_api->CreateStatus(
        ORT_FAIL, "ORT host does not expose the Plugin Execution Provider API");
  }

  // Manual init for the C++ API
  Ort::InitApi(ort_api);

  try {
    std::unique_ptr<OrtEpFactory> factory =
        std::make_unique<MusaEpFactory>(*ort_api, *ep_api, *default_logger);

    factories[0] = factory.release();
    *num_factories = 1;
  } catch (const std::exception& ex) {
    return ort_api->CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
  } catch (...) {
    return ort_api->CreateStatus(
        ORT_RUNTIME_EXCEPTION,
        "Unknown exception while creating MUSAExecutionProvider factory");
  }

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<MusaEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
