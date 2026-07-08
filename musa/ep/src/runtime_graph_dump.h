// Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

bool RuntimeGraphDumpEnabled();

struct RuntimeGraphNodeMetadata {
  std::string kind;
  std::string display_type;
  std::string node_name;
  std::string domain_name;
  int since_version = 0;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::string> source_nodes;
  std::vector<std::string> source_ops;
};

void RegisterRuntimeKernelInstance(const void* impl,
                                   RuntimeGraphNodeMetadata metadata);
void UnregisterRuntimeKernelInstance(const void* impl);

void RegisterRuntimeFusionInstance(const void* impl,
                                   RuntimeGraphNodeMetadata metadata);

uint64_t BeginRuntimeCompute(const void* impl, const char* fallback_label);
void EndRuntimeCompute(uint64_t compute_id);
