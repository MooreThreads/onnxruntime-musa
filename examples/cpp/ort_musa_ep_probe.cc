// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <onnxruntime_cxx_api.h>

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kMusaEpName = "MUSAExecutionProvider";

void PrintUsage(const char* program) {
  std::cerr << "usage: " << program << " MUSA_PLUGIN_SO [MODEL]\n"
            << "  Without MODEL: register the plugin and enumerate devices.\n"
            << "  With MODEL: also create a MUSA-only inference session.\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    PrintUsage(argv[0]);
    return 2;
  }

  try {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ort-musa-cxx-probe"};
    env.RegisterExecutionProviderLibrary(kMusaEpName, argv[1]);

    std::vector<Ort::ConstEpDevice> selected_devices;
    for (const Ort::ConstEpDevice& device : env.GetEpDevices()) {
      if (std::string{device.EpName()} == kMusaEpName) {
        selected_devices.push_back(device);
      }
    }

    if (selected_devices.empty()) {
      std::cerr << "registered " << kMusaEpName
                << " but it advertised no visible MUSA devices\n";
      return 1;
    }

    std::cout << "registered_ep=" << kMusaEpName
              << " visible_devices=" << selected_devices.size() << "\n";

    if (argc == 3) {
      Ort::SessionOptions session_options;
      session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      session_options.AppendExecutionProvider_V2(
          env, selected_devices,
          std::unordered_map<std::string, std::string>{});
      Ort::Session session{env, argv[2], session_options};
      std::cout << "session_created model=" << argv[2] << "\n";
    }
  } catch (const Ort::Exception& ex) {
    std::cerr << "ORT error: " << ex.what() << " (code=" << ex.GetOrtErrorCode()
              << ")\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
