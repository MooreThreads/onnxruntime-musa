#pragma once

#include <sstream>
#include <stdexcept>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define RETURN_IF_ERROR(expr)    \
  do {                           \
    Ort::Status _status{(expr)}; \
    if (!_status.IsOK()) {       \
      return _status.release();  \
    }                            \
  } while (0)

#define RETURN_IF(condition, api, message)             \
  do {                                                 \
    if (condition) {                                   \
      return (api).CreateStatus(ORT_EP_FAIL, message); \
    }                                                  \
  } while (0)

#define EXCEPTION_TO_STATUS_BEGIN try {
#define EXCEPTION_TO_STATUS_END                                     \
  }                                                                 \
  catch (const Ort::Exception& ex) {                                \
    Ort::Status status(ex);                                         \
    return status.release();                                        \
  }                                                                 \
  catch (const std::exception& ex) {                                \
    Ort::Status status(ex.what(), ORT_EP_FAIL);                     \
    return status.release();                                        \
  }                                                                 \
  catch (...) {                                                     \
    Ort::Status status("Unknown plugin EP exception", ORT_EP_FAIL); \
    return status.release();                                        \
  }

#define IGNORE_ORT_STATUS(expr)    \
  do {                             \
    OrtStatus* _status = (expr);   \
    Ort::Status _ignored{_status}; \
  } while (0)

#ifdef _WIN32
#define ORT_MUSA_FILE __FILE__
#else
#define ORT_MUSA_FILE __FILE__
#endif

inline OrtStatus* GetSessionConfigEntryOrDefault(
    const OrtSessionOptions& session_options, const char* key,
    const std::string& default_value, std::string& value) {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstSessionOptions options{&session_options};
  value = options.GetConfigEntryOrDefault(key, default_value);
  return nullptr;
  EXCEPTION_TO_STATUS_END
}
