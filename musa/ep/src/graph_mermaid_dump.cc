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

#include "graph_mermaid_dump.h"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "fusion/fusion_dtype.h"

namespace {

constexpr const char* kDumpGetCapabilityGraphMermaidEnv =
    "ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID";
constexpr const char* kDumpGetCapabilityGraphMermaidPathEnv =
    "ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID_PATH";

std::atomic<uint64_t> g_get_capability_dump_counter{0};

bool EnvEnabled(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }

  std::string normalized;
  for (const char* p = value; *p != '\0'; ++p) {
    normalized.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
  }

  return normalized != "0" && normalized != "false" && normalized != "off" &&
         normalized != "no";
}

std::string EscapeLabel(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '&':
        out += "&amp;";
        break;
      case '"':
        out += "'";
        break;
      case '\n':
      case '\r':
        out += ' ';
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

std::string NodeId(const Ort::ConstNode& node) {
  return "node_" + std::to_string(node.GetId());
}

std::string NodeId(const std::string& prefix, const Ort::ConstNode& node) {
  return prefix + "_" + NodeId(node);
}

std::string EdgeKey(const std::string& from, const std::string& to,
                    const std::string& label) {
  return from + "\n" + to + "\n" + label;
}

std::string NumberedPath(const char* configured_path,
                         const char* default_prefix, uint64_t index) {
  if (configured_path == nullptr || configured_path[0] == '\0') {
    return std::string(default_prefix) + "_" + std::to_string(index) + ".mmd";
  }

  std::string path(configured_path);
  const std::string token = "{}";
  const size_t token_pos = path.find(token);
  if (token_pos != std::string::npos) {
    path.replace(token_pos, token.size(), std::to_string(index));
    return path;
  }

  if (index == 0) {
    return path;
  }

  const size_t slash_pos = path.find_last_of("/\\");
  const size_t dot_pos = path.find_last_of('.');
  if (dot_pos != std::string::npos &&
      (slash_pos == std::string::npos || dot_pos > slash_pos)) {
    return path.substr(0, dot_pos) + "_" + std::to_string(index) +
           path.substr(dot_pos);
  }

  return path + "_" + std::to_string(index);
}

std::string DTypeLabel(Ort::ConstValueInfo value_info) {
  try {
    auto elem_type = musa_ep::GetTensorElementType(value_info);
    if (elem_type.has_value()) {
      return musa_ep::FusionDTypeName(*elem_type);
    }
  } catch (...) {
  }
  return "dtype=?";
}

std::string ValueLabel(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr) {
    return "";
  }
  return DTypeLabel(value_info);
}

void WriteEdge(std::ostream& out, std::unordered_set<std::string>& seen,
               const std::string& from, const std::string& to,
               Ort::ConstValueInfo value_info) {
  const std::string label = ValueLabel(value_info);
  const std::string key = EdgeKey(from, to, label);
  if (!seen.insert(key).second) {
    return;
  }
  out << "  " << from << " -->";
  if (!label.empty()) {
    out << "|\"" << EscapeLabel(label) << "\"|";
  }
  out << " " << to << "\n";
}

std::string NodeLabel(const Ort::ConstNode& node) {
  std::string label = node.GetOperatorType();
  const std::string node_name = node.GetName();
  if (!node_name.empty()) {
    label += " | ";
    label += node_name;
  }
  return label;
}

void WriteGraph(std::ostream& out, const OrtGraph& ort_graph,
                const std::string& node_prefix,
                std::unordered_set<std::string>& seen_edges) {
  Ort::ConstGraph graph{&ort_graph};
  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();

  std::map<std::string, Ort::ConstValueInfo> external_values;
  std::unordered_set<std::string> produced_values;
  std::unordered_set<std::string> consumed_values;
  for (const Ort::ConstNode& node : nodes) {
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      if (input != nullptr && !input.GetName().empty()) {
        consumed_values.insert(input.GetName());
      }
    }
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output != nullptr && !output.GetName().empty()) {
        produced_values.insert(output.GetName());
      }
    }
  }

  for (const Ort::ConstNode& node : nodes) {
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      if (input != nullptr && !input.GetName().empty() &&
          produced_values.count(input.GetName()) == 0) {
        external_values.emplace(input.GetName(), input);
      }
    }
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output != nullptr && !output.GetName().empty() &&
          consumed_values.count(output.GetName()) == 0) {
        external_values.emplace(output.GetName(), output);
      }
    }
  }

  uint64_t value_index = 0;
  std::map<std::string, std::string> external_value_ids;
  for (const auto& [value_name, value_info] : external_values) {
    const std::string value_id =
        node_prefix + "_value_" + std::to_string(value_index++);
    external_value_ids.emplace(value_name, value_id);
    out << "    " << value_id << "[\"" << EscapeLabel(ValueLabel(value_info))
        << "\"]\n";
  }

  for (const Ort::ConstNode& node : nodes) {
    out << "    " << NodeId(node_prefix, node) << "[\""
        << EscapeLabel(NodeLabel(node)) << "\"]\n";
  }

  for (const Ort::ConstNode& node : nodes) {
    const std::string to = NodeId(node_prefix, node);
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      if (input == nullptr) {
        continue;
      }

      Ort::ValueInfoConsumerProducerInfo producer = input.GetProducerNode();
      if (producer.node != nullptr) {
        WriteEdge(out, seen_edges, NodeId(node_prefix, producer.node), to,
                  input);
        continue;
      }

      auto external_it = external_value_ids.find(input.GetName());
      if (external_it != external_value_ids.end()) {
        WriteEdge(out, seen_edges, external_it->second, to, input);
      }
    }

    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output == nullptr || output.GetName().empty() ||
          consumed_values.count(output.GetName()) != 0) {
        continue;
      }

      auto external_it = external_value_ids.find(output.GetName());
      if (external_it != external_value_ids.end()) {
        WriteEdge(out, seen_edges, to, external_it->second, output);
      }
    }
  }
}

}  // namespace

Ort::Status DumpGraphToMermaidIfEnabled(const OrtGraph& ort_graph,
                                        const char* enabled_env,
                                        const char* path_env,
                                        const char* default_prefix,
                                        std::atomic<uint64_t>& counter) {
  const char* enabled = std::getenv(enabled_env);
  if (!EnvEnabled(enabled)) {
    return Ort::Status(nullptr);
  }

  const uint64_t dump_index = counter.fetch_add(1);
  const std::string path =
      NumberedPath(std::getenv(path_env), default_prefix, dump_index);
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    const std::string message =
        "Failed to open MUSA EP Mermaid graph dump file: " + path;
    return Ort::Status(message.c_str(), ORT_EP_FAIL);
  }

  out << "flowchart TD\n";

  std::unordered_set<std::string> seen_edges;
  WriteGraph(out, ort_graph, "graph", seen_edges);

  if (!out) {
    const std::string message =
        "Failed while writing MUSA EP Mermaid graph dump file: " + path;
    return Ort::Status(message.c_str(), ORT_EP_FAIL);
  }

  return Ort::Status(nullptr);
}

Ort::Status DumpGetCapabilityGraphToMermaidIfEnabled(
    const OrtGraph& ort_graph) {
  return DumpGraphToMermaidIfEnabled(
      ort_graph, kDumpGetCapabilityGraphMermaidEnv,
      kDumpGetCapabilityGraphMermaidPathEnv, "musa_ep_get_capability_graph",
      g_get_capability_dump_counter);
}
