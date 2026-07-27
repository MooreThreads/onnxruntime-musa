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
