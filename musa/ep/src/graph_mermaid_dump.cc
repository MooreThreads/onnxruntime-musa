// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "graph_mermaid_dump.h"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr const char* kDumpGraphMermaidEnv = "ORT_MUSA_DUMP_GRAPH_MERMAID";
constexpr const char* kDumpGraphMermaidPathEnv =
    "ORT_MUSA_DUMP_GRAPH_MERMAID_PATH";

std::atomic<uint64_t> g_dump_counter{0};

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

std::string EdgeKey(const std::string& from, const std::string& to) {
  return from + "\n" + to;
}

std::string NumberedPath(const char* configured_path, uint64_t index) {
  if (configured_path == nullptr || configured_path[0] == '\0') {
    return "musa_ep_getcapability_graph_" + std::to_string(index) + ".mmd";
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

void WriteEdge(std::ostream& out, std::unordered_set<std::string>& seen,
               const std::string& from, const std::string& to) {
  const std::string key = EdgeKey(from, to);
  if (!seen.insert(key).second) {
    return;
  }
  out << "  " << from << " --> " << to << "\n";
}

std::string NodeLabel(const Ort::ConstNode& node) {
  return node.GetOperatorType();
}

}  // namespace

Ort::Status DumpGraphToMermaidIfEnabled(const OrtGraph& ort_graph) {
  const char* enabled = std::getenv(kDumpGraphMermaidEnv);
  if (!EnvEnabled(enabled)) {
    return Ort::Status(nullptr);
  }

  const uint64_t dump_index = g_dump_counter.fetch_add(1);
  const std::string path =
      NumberedPath(std::getenv(kDumpGraphMermaidPathEnv), dump_index);
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    const std::string message =
        "Failed to open MUSA EP Mermaid graph dump file: " + path;
    return Ort::Status(message.c_str(), ORT_EP_FAIL);
  }

  Ort::ConstGraph graph{&ort_graph};
  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();

  out << "flowchart TD\n";

  std::unordered_set<std::string> seen_edges;

  for (const Ort::ConstNode& node : nodes) {
    out << "  " << NodeId(node) << "[\"" << EscapeLabel(NodeLabel(node))
        << "\"]\n";
  }

  for (const Ort::ConstNode& node : nodes) {
    const std::string to = NodeId(node);
    for (Ort::ConstValueInfo input : node.GetInputs()) {
      if (input == nullptr) {
        continue;
      }

      Ort::ValueInfoConsumerProducerInfo producer = input.GetProducerNode();
      if (producer.node != nullptr) {
        WriteEdge(out, seen_edges, NodeId(producer.node), to);
      }
    }
  }

  if (!out) {
    const std::string message =
        "Failed while writing MUSA EP Mermaid graph dump file: " + path;
    return Ort::Status(message.c_str(), ORT_EP_FAIL);
  }

  return Ort::Status(nullptr);
}
