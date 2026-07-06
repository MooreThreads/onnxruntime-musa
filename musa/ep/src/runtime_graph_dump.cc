// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "runtime_graph_dump.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr const char* kDumpRuntimeGraphMermaidEnv =
    "ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID";
constexpr const char* kDumpRuntimeGraphMermaidPathEnv =
    "ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID_PATH";

struct RuntimeExecNode {
  uint64_t id = 0;
  RuntimeGraphNodeMetadata metadata;
  uint64_t run_count = 0;
  uint64_t first_run_order = 0;
};

struct RuntimeGraphState {
  std::mutex mutex;
  std::unordered_map<const void*, uint64_t> node_ids_by_impl;
  std::vector<RuntimeExecNode> nodes;
  std::atomic<uint64_t> next_node_id{1};
  std::atomic<uint64_t> next_run_order{1};
};

RuntimeGraphState& State() {
  static RuntimeGraphState* state = new RuntimeGraphState();
  return *state;
}

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

std::string OutputPath() {
  const char* configured = std::getenv(kDumpRuntimeGraphMermaidPathEnv);
  if (configured == nullptr || configured[0] == '\0') {
    return "musa_ep_runtime_execution_graph.mmd";
  }
  return configured;
}

std::string NodeMermaidId(uint64_t id) { return "exec_" + std::to_string(id); }

std::string ValueMermaidId(uint64_t index) {
  return "value_" + std::to_string(index);
}

std::string JoinUnique(const std::vector<std::string>& values,
                       const char* separator) {
  std::vector<std::string> unique_values;
  std::unordered_set<std::string> seen;
  for (const std::string& value : values) {
    if (!value.empty() && seen.insert(value).second) {
      unique_values.push_back(value);
    }
  }

  std::string joined;
  for (size_t i = 0; i < unique_values.size(); ++i) {
    if (i != 0) {
      joined += separator;
    }
    joined += unique_values[i];
  }
  return joined;
}

std::string NodeLabel(const RuntimeExecNode& node) {
  std::string label = node.metadata.display_type.empty()
                          ? "MUSA execution node"
                          : node.metadata.display_type;
  if (!node.metadata.node_name.empty()) {
    label += " | ";
    label += node.metadata.node_name;
  }
  if (!node.metadata.domain_name.empty()) {
    label += "<br/>";
    label += node.metadata.domain_name;
  }
  if (!node.metadata.kind.empty()) {
    label += "<br/>";
    label += node.metadata.kind;
  }
  const std::string source_ops = JoinUnique(node.metadata.source_ops, " + ");
  if (!source_ops.empty()) {
    label += "<br/>source: ";
    label += source_ops;
  }
  label += "<br/>runs=" + std::to_string(node.run_count);
  return label;
}

void WriteEdge(std::ostream& out, std::unordered_set<std::string>& seen_edges,
               const std::string& from, const std::string& to,
               const std::string& value_name) {
  const std::string key = from + "\n" + to + "\n" + value_name;
  if (!seen_edges.insert(key).second) {
    return;
  }

  out << "  " << from << " -->";
  if (!value_name.empty()) {
    out << "|\"" << EscapeLabel(value_name) << "\"|";
  }
  out << " " << to << "\n";
}

uint64_t FindProducerForInput(
    const std::unordered_map<std::string, std::vector<uint64_t>>&
        producers_by_value,
    const std::unordered_map<uint64_t, const RuntimeExecNode*>& nodes_by_id,
    const RuntimeExecNode& consumer, const std::string& input) {
  auto producer_it = producers_by_value.find(input);
  if (producer_it == producers_by_value.end()) {
    return 0;
  }

  const std::vector<uint64_t>& producers = producer_it->second;
  if (producers.size() == 1) {
    return producers[0];
  }

  uint64_t selected_id = 0;
  uint64_t selected_order = 0;
  for (uint64_t producer_id : producers) {
    auto node_it = nodes_by_id.find(producer_id);
    if (node_it == nodes_by_id.end()) {
      continue;
    }

    const RuntimeExecNode* producer = node_it->second;
    if (consumer.first_run_order == 0 || producer->first_run_order == 0 ||
        producer->first_run_order >= consumer.first_run_order) {
      continue;
    }

    if (producer->first_run_order >= selected_order) {
      selected_id = producer_id;
      selected_order = producer->first_run_order;
    }
  }

  return selected_id;
}

void DumpRuntimeGraphAtExit() {
  if (!RuntimeGraphDumpEnabled()) {
    return;
  }

  std::vector<RuntimeExecNode> nodes;
  {
    RuntimeGraphState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    nodes = state.nodes;
  }

  std::ofstream out(OutputPath(), std::ios::out | std::ios::trunc);
  if (!out) {
    return;
  }

  out << "flowchart TD\n";

  std::unordered_map<std::string, std::vector<uint64_t>> producers_by_value;
  std::unordered_set<std::string> consumed_values;
  std::unordered_map<uint64_t, const RuntimeExecNode*> nodes_by_id;
  for (const RuntimeExecNode& node : nodes) {
    nodes_by_id.emplace(node.id, &node);
    for (const std::string& output : node.metadata.outputs) {
      if (!output.empty()) {
        producers_by_value[output].push_back(node.id);
      }
    }
    for (const std::string& input : node.metadata.inputs) {
      if (!input.empty()) {
        consumed_values.insert(input);
      }
    }
  }

  std::map<std::string, uint64_t> external_values;
  for (const RuntimeExecNode& node : nodes) {
    for (const std::string& input : node.metadata.inputs) {
      const uint64_t producer_id =
          FindProducerForInput(producers_by_value, nodes_by_id, node, input);
      if (!input.empty() && producer_id == 0) {
        external_values.emplace(input, external_values.size());
      }
    }
    for (const std::string& output : node.metadata.outputs) {
      if (!output.empty() && consumed_values.count(output) == 0) {
        external_values.emplace(output, external_values.size());
      }
    }
  }

  for (const auto& [value_name, value_index] : external_values) {
    out << "  " << ValueMermaidId(value_index) << "[\""
        << EscapeLabel(value_name) << "\"]\n";
  }

  for (const RuntimeExecNode& node : nodes) {
    out << "  " << NodeMermaidId(node.id) << "[\""
        << EscapeLabel(NodeLabel(node)) << "\"]\n";
  }

  std::unordered_set<std::string> seen_edges;
  for (const RuntimeExecNode& node : nodes) {
    const std::string to = NodeMermaidId(node.id);
    for (const std::string& input : node.metadata.inputs) {
      if (input.empty()) {
        continue;
      }
      const uint64_t producer_id =
          FindProducerForInput(producers_by_value, nodes_by_id, node, input);
      if (producer_id != 0) {
        WriteEdge(out, seen_edges, NodeMermaidId(producer_id), to, input);
        continue;
      }

      auto external_it = external_values.find(input);
      if (external_it != external_values.end()) {
        WriteEdge(out, seen_edges, ValueMermaidId(external_it->second), to,
                  input);
      }
    }

    for (const std::string& output : node.metadata.outputs) {
      if (output.empty() || consumed_values.count(output) != 0) {
        continue;
      }
      auto external_it = external_values.find(output);
      if (external_it != external_values.end()) {
        WriteEdge(out, seen_edges, to, ValueMermaidId(external_it->second),
                  output);
      }
    }
  }
}

void RegisterDumpAtExitOnce() {
  static const bool registered = []() {
    std::atexit(DumpRuntimeGraphAtExit);
    return true;
  }();
  (void)registered;
}

}  // namespace

bool RuntimeGraphDumpEnabled() {
  static const bool enabled =
      EnvEnabled(std::getenv(kDumpRuntimeGraphMermaidEnv));
  if (enabled) {
    RegisterDumpAtExitOnce();
  }
  return enabled;
}

void RegisterRuntimeKernelInstance(const void* impl,
                                   RuntimeGraphNodeMetadata metadata) {
  if (!RuntimeGraphDumpEnabled() || impl == nullptr) {
    return;
  }

  if (metadata.kind.empty()) {
    metadata.kind = "kernel";
  }

  RuntimeGraphState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  const uint64_t id = state.next_node_id.fetch_add(1);
  state.node_ids_by_impl[impl] = id;
  state.nodes.push_back({id, std::move(metadata), 0});
}

void UnregisterRuntimeKernelInstance(const void* impl) {
  if (!RuntimeGraphDumpEnabled() || impl == nullptr) {
    return;
  }

  RuntimeGraphState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.node_ids_by_impl.erase(impl);
}

void RegisterRuntimeFusionInstance(const void* impl,
                                   RuntimeGraphNodeMetadata metadata) {
  if (!RuntimeGraphDumpEnabled() || impl == nullptr) {
    return;
  }

  if (metadata.kind.empty()) {
    metadata.kind = "fusion";
  }

  RuntimeGraphState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  const uint64_t id = state.next_node_id.fetch_add(1);
  state.node_ids_by_impl[impl] = id;
  state.nodes.push_back({id, std::move(metadata), 0});
}

uint64_t BeginRuntimeCompute(const void* impl, const char* fallback_label) {
  if (!RuntimeGraphDumpEnabled()) {
    return 0;
  }

  RuntimeGraphState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto id_it = state.node_ids_by_impl.find(impl);
  if (id_it != state.node_ids_by_impl.end()) {
    for (RuntimeExecNode& node : state.nodes) {
      if (node.id == id_it->second) {
        if (node.run_count == 0) {
          node.first_run_order = state.next_run_order.fetch_add(1);
        }
        ++node.run_count;
        return node.id;
      }
    }
  }

  RuntimeGraphNodeMetadata metadata;
  metadata.kind = "unknown";
  metadata.display_type =
      fallback_label == nullptr ? "Compute" : fallback_label;
  const uint64_t id = state.next_node_id.fetch_add(1);
  state.node_ids_by_impl[impl] = id;
  RuntimeExecNode node;
  node.id = id;
  node.metadata = std::move(metadata);
  node.run_count = 1;
  node.first_run_order = state.next_run_order.fetch_add(1);
  state.nodes.push_back(std::move(node));
  return id;
}

void EndRuntimeCompute(uint64_t compute_id) { (void)compute_id; }
