#include "fusion/target_id_count_embedding_fusion.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "kernels/tensor/target_id_count_embedding_impl.h"

namespace {

std::unordered_map<std::string, Ort::ConstNode> ProducersInGraph(
    Ort::ConstGraph graph) {
  std::unordered_map<std::string, Ort::ConstNode> producers;
  for (Ort::ConstNode node : graph.GetNodes()) {
    for (Ort::ConstValueInfo output : node.GetOutputs()) {
      if (output != nullptr) {
        producers.emplace(musa_ep::Name(output), node);
      }
    }
  }
  return producers;
}

std::unordered_map<std::string, size_t> ValueIndices(
    const std::vector<Ort::ConstValueInfo>& values) {
  std::unordered_map<std::string, size_t> indices;
  for (size_t i = 0; i < values.size(); ++i) {
    indices.emplace(musa_ep::Name(values[i]), i);
  }
  return indices;
}

size_t MappedIndex(const std::unordered_map<std::string, size_t>& indices,
                   const std::string& name, const char* role) {
  auto it = indices.find(name);
  if (it == indices.end()) {
    throw std::runtime_error(std::string("TargetIdCountEmbedding missing ") +
                             role + " " + name);
  }
  return it->second;
}

Ort::ConstNode ProducerInGraph(
    const std::unordered_map<std::string, Ort::ConstNode>& producers,
    Ort::ConstValueInfo value_info, const char* op_type) {
  auto it = producers.find(musa_ep::Name(value_info));
  if (it == producers.end() || !musa_ep::IsOnnxOp(it->second, op_type)) {
    return Ort::ConstNode{nullptr};
  }
  return it->second;
}

int64_t ReadRequiredScalarIntInitializer(Ort::ConstValueInfo value_info,
                                         const char* name) {
  std::optional<int64_t> value = musa_ep::ReadScalarIntInitializer(value_info);
  if (!value.has_value()) {
    throw std::runtime_error(std::string("TargetIdCountEmbedding ") + name +
                             " must be a scalar int initializer");
  }
  return *value;
}

std::optional<int64_t> ReadOptionalScalarIntInitializer(
    Ort::ConstValueInfo value_info) {
  return musa_ep::ReadScalarIntInitializer(value_info);
}

std::vector<int64_t> ReadRequiredAxes(Ort::ConstValueInfo value_info,
                                      const char* name) {
  std::optional<std::vector<int64_t>> axes =
      musa_ep::ReadSmallIntInitializer(value_info);
  if (!axes.has_value()) {
    throw std::runtime_error(std::string("TargetIdCountEmbedding ") + name +
                             " axes must be an int initializer");
  }
  return *axes;
}

std::vector<int64_t> ApplyReduceSumShape(std::vector<int64_t> input_shape,
                                         int64_t axis, bool keepdims) {
  int64_t normalized_axis = 0;
  if (!musa_ep::NormalizeAxis(axis, input_shape.size(), normalized_axis)) {
    throw std::runtime_error("TargetIdCountEmbedding invalid reduce axis");
  }
  if (keepdims) {
    input_shape[static_cast<size_t>(normalized_axis)] = 1;
  } else {
    input_shape.erase(input_shape.begin() + normalized_axis);
  }
  return input_shape;
}

std::vector<int64_t> ApplyUnsqueezeShape(std::vector<int64_t> input_shape,
                                         std::vector<int64_t> axes) {
  const int64_t output_rank =
      static_cast<int64_t>(input_shape.size() + axes.size());
  for (int64_t& axis : axes) {
    if (axis < 0) {
      axis += output_rank;
    }
    if (axis < 0 || axis >= output_rank) {
      throw std::runtime_error("TargetIdCountEmbedding invalid unsqueeze axis");
    }
  }
  std::sort(axes.begin(), axes.end());
  std::vector<int64_t> output_shape;
  output_shape.reserve(static_cast<size_t>(output_rank));
  size_t input_index = 0;
  size_t axes_index = 0;
  for (int64_t dim = 0; dim < output_rank; ++dim) {
    if (axes_index < axes.size() && axes[axes_index] == dim) {
      output_shape.push_back(1);
      ++axes_index;
    } else {
      output_shape.push_back(input_shape[input_index++]);
    }
  }
  return output_shape;
}

struct TargetIdCountEmbeddingValues {
  Ort::ConstValueInfo ids{nullptr};
  Ort::ConstValueInfo target{nullptr};
  Ort::ConstValueInfo table{nullptr};
  Ort::ConstValueInfo embedding_output{nullptr};
  Ort::ConstValueInfo count_output{nullptr};
  int64_t pad = 0;
  int64_t cap = 0;
  int64_t reduce_axis = -1;
  bool reduce_keepdims = false;
  std::vector<int64_t> source_unsqueeze_axes;
  std::vector<int64_t> bucket_unsqueeze_axes;
  std::vector<int64_t> count_unsqueeze_axes;
  std::optional<int64_t> target_constant;
};

bool ResolveTargetIdCountEmbeddingValues(
    Ort::ConstGraph graph, TargetIdCountEmbeddingValues& resolved) {
  auto producers = ProducersInGraph(graph);
  for (Ort::ConstNode gather : graph.GetNodes()) {
    if (!musa_ep::IsOnnxOp(gather, "Gather")) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> gather_inputs = gather.GetInputs();
    std::vector<Ort::ConstValueInfo> gather_outputs = gather.GetOutputs();
    if (gather_inputs.size() != 2 || gather_outputs.size() != 1) {
      continue;
    }

    Ort::ConstNode bucket_unsqueeze =
        ProducerInGraph(producers, gather_inputs[1], "Unsqueeze");
    if (!bucket_unsqueeze) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> bucket_inputs =
        bucket_unsqueeze.GetInputs();
    if (bucket_inputs.size() != 2) {
      continue;
    }
    Ort::ConstNode min_node =
        ProducerInGraph(producers, bucket_inputs[0], "Min");
    if (!min_node) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> min_inputs = min_node.GetInputs();
    if (min_inputs.size() != 2) {
      continue;
    }

    Ort::ConstNode reduce{nullptr};
    Ort::ConstValueInfo cap_input{nullptr};
    for (Ort::ConstValueInfo input : min_inputs) {
      Ort::ConstNode producer = ProducerInGraph(producers, input, "ReduceSum");
      if (producer) {
        reduce = producer;
      } else {
        cap_input = input;
      }
    }
    if (!reduce) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> reduce_inputs = reduce.GetInputs();
    std::vector<Ort::ConstValueInfo> reduce_outputs = reduce.GetOutputs();
    if (reduce_inputs.size() != 2 || reduce_outputs.size() != 1) {
      continue;
    }

    Ort::ConstNode count_unsqueeze{nullptr};
    Ort::ConstNode count_cast{nullptr};
    for (const auto& consumer : reduce_outputs[0].GetConsumers()) {
      if (consumer.node.GetId() == min_node.GetId() ||
          !musa_ep::IsOnnxOp(consumer.node, "Unsqueeze")) {
        continue;
      }
      std::vector<Ort::ConstValueInfo> outputs = consumer.node.GetOutputs();
      if (outputs.size() != 1) {
        continue;
      }
      for (const auto& cast_consumer : outputs[0].GetConsumers()) {
        if (musa_ep::IsOnnxOp(cast_consumer.node, "Cast")) {
          count_unsqueeze = consumer.node;
          count_cast = cast_consumer.node;
          break;
        }
      }
    }
    if (!count_unsqueeze || !count_cast) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> count_outputs = count_cast.GetOutputs();
    if (count_outputs.size() != 1) {
      continue;
    }

    Ort::ConstNode mask_cast =
        ProducerInGraph(producers, reduce_inputs[0], "Cast");
    if (!mask_cast) {
      continue;
    }
    std::vector<Ort::ConstValueInfo> mask_cast_inputs = mask_cast.GetInputs();
    if (mask_cast_inputs.size() != 1) {
      continue;
    }
    Ort::ConstNode and_node =
        ProducerInGraph(producers, mask_cast_inputs[0], "And");
    if (!and_node) {
      continue;
    }

    Ort::ConstNode hit_equal{nullptr};
    Ort::ConstNode valid_unsqueeze{nullptr};
    for (Ort::ConstValueInfo input : and_node.GetInputs()) {
      Ort::ConstNode producer = ProducerInGraph(producers, input, "Equal");
      if (musa_ep::IsOnnxOp(producer, "Equal")) {
        hit_equal = producer;
        continue;
      }
      producer = ProducerInGraph(producers, input, "Unsqueeze");
      if (musa_ep::IsOnnxOp(producer, "Unsqueeze")) {
        valid_unsqueeze = producer;
      }
    }
    if (!hit_equal || !valid_unsqueeze) {
      continue;
    }

    Ort::ConstNode sub{nullptr};
    Ort::ConstValueInfo zero_input{nullptr};
    for (Ort::ConstValueInfo input : hit_equal.GetInputs()) {
      Ort::ConstNode producer = ProducerInGraph(producers, input, "Sub");
      if (musa_ep::IsOnnxOp(producer, "Sub")) {
        sub = producer;
      } else {
        zero_input = input;
      }
    }
    if (!sub || ReadRequiredScalarIntInitializer(zero_input, "zero") != 0) {
      continue;
    }

    Ort::ConstValueInfo ids{nullptr};
    Ort::ConstValueInfo target{nullptr};
    Ort::ConstNode source_unsqueeze{nullptr};
    for (Ort::ConstValueInfo input : sub.GetInputs()) {
      Ort::ConstNode unsqueeze = ProducerInGraph(producers, input, "Unsqueeze");
      if (musa_ep::IsOnnxOp(unsqueeze, "Unsqueeze")) {
        std::vector<Ort::ConstValueInfo> unsqueeze_inputs =
            unsqueeze.GetInputs();
        if (!unsqueeze_inputs.empty() &&
            musa_ep::IsIntTensorValueInfo(unsqueeze_inputs[0])) {
          ids = unsqueeze_inputs[0];
          source_unsqueeze = unsqueeze;
        } else {
          target = input;
        }
      } else {
        target = input;
      }
    }
    if (!ids || !target) {
      continue;
    }

    Ort::ConstNode not_node =
        ProducerInGraph(producers, valid_unsqueeze.GetInputs()[0], "Not");
    if (!not_node || not_node.GetInputs().size() != 1) {
      continue;
    }
    Ort::ConstNode pad_equal =
        ProducerInGraph(producers, not_node.GetInputs()[0], "Equal");
    if (!pad_equal) {
      continue;
    }
    Ort::ConstValueInfo pad_input{nullptr};
    bool has_ids = false;
    for (Ort::ConstValueInfo input : pad_equal.GetInputs()) {
      if (musa_ep::Name(input) == musa_ep::Name(ids)) {
        has_ids = true;
      } else {
        pad_input = input;
      }
    }
    if (!has_ids || !pad_input) {
      continue;
    }

    resolved.ids = ids;
    resolved.target = target;
    resolved.table = gather_inputs[0];
    resolved.embedding_output = gather_outputs[0];
    resolved.count_output = count_outputs[0];
    resolved.pad = ReadRequiredScalarIntInitializer(pad_input, "pad");
    resolved.cap = ReadRequiredScalarIntInitializer(cap_input, "cap");
    resolved.reduce_axis = ReadRequiredAxes(reduce_inputs[1], "reduce")[0];
    resolved.reduce_keepdims =
        musa_ep::GetIntAttribute(reduce, "keepdims").value_or(1) != 0;
    resolved.source_unsqueeze_axes =
        ReadRequiredAxes(source_unsqueeze.GetInputs()[1], "source unsqueeze");
    resolved.bucket_unsqueeze_axes =
        ReadRequiredAxes(bucket_inputs[1], "bucket unsqueeze");
    resolved.count_unsqueeze_axes =
        ReadRequiredAxes(count_unsqueeze.GetInputs()[1], "count unsqueeze");
    resolved.target_constant = ReadOptionalScalarIntInitializer(target);
    return true;
  }
  return false;
}

struct TargetIdCountEmbeddingFusionCompute : FusionNodeCompute {
  TargetIdCountEmbeddingFusionCompute(TargetIdCountEmbeddingValues values,
                                      std::optional<size_t> target_index,
                                      size_t ids_index, size_t table_index,
                                      size_t embedding_output_index,
                                      size_t count_output_index)
      : values(std::move(values)),
        target_index(target_index),
        ids_index(ids_index),
        table_index(table_index),
        embedding_output_index(embedding_output_index),
        count_output_index(count_output_index) {}

  OrtStatus* Compute(OrtKernelContext* kernel_context) const override {
    try {
      Ort::KernelContext ctx(kernel_context);
      Ort::ConstValue ids = ctx.GetInput(ids_index);
      Ort::ConstValue table = ctx.GetInput(table_index);
      Ort::ConstValue target{nullptr};
      if (target_index.has_value()) {
        target = ctx.GetInput(*target_index);
      }

      auto ids_info = ids.GetTensorTypeAndShapeInfo();
      auto table_info = table.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> ids_shape = ids_info.GetShape();
      std::vector<int64_t> table_shape = table_info.GetShape();
      if (ids_shape.empty() || table_shape.size() < 2 || table_shape[0] <= 0) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "TargetIdCountEmbedding requires rank-1+ ids and rank-2+ table");
      }
      if (ids_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
          ids_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "TargetIdCountEmbedding unsupported ids dtype");
      }
      if (table_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "TargetIdCountEmbedding only supports float embedding table");
      }

      std::vector<int64_t> source_shape =
          ApplyUnsqueezeShape(ids_shape, values.source_unsqueeze_axes);
      std::vector<int64_t> reduced_shape = ApplyReduceSumShape(
          source_shape, values.reduce_axis, values.reduce_keepdims);
      const int64_t rows = NumElements(reduced_shape);
      const int64_t ids_count = ids_info.GetElementCount();
      if (rows < 0 || ids_count < 0 || rows == 0 || ids_count % rows != 0) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "TargetIdCountEmbedding cannot derive sequence length");
      }
      const int64_t sequence_length = ids_count / rows;
      int64_t embedding_dim = 1;
      for (size_t i = 1; i < table_shape.size(); ++i) {
        embedding_dim *= table_shape[i];
      }

      std::vector<int64_t> count_shape =
          ApplyUnsqueezeShape(reduced_shape, values.count_unsqueeze_axes);
      std::vector<int64_t> bucket_shape =
          ApplyUnsqueezeShape(reduced_shape, values.bucket_unsqueeze_axes);
      std::vector<int64_t> embedding_shape = bucket_shape;
      embedding_shape.insert(embedding_shape.end(), table_shape.begin() + 1,
                             table_shape.end());

      Ort::UnownedValue embedding_output =
          ctx.GetOutput(embedding_output_index, embedding_shape);
      Ort::UnownedValue count_output =
          ctx.GetOutput(count_output_index, count_shape);
      if (!IsGpuMemory(embedding_output.GetTensorMemoryInfo()) ||
          !IsGpuMemory(count_output.GetTensorMemoryInfo())) {
        return Ort::GetApi().CreateStatus(
            ORT_NOT_IMPLEMENTED,
            "TargetIdCountEmbedding requires MUSA outputs");
      }

      const void* target_data = nullptr;
      int64_t target_count = 0;
      int64_t target_elem_size = 0;
      DeviceInputBuffer target_buffer;
      if (target_index.has_value()) {
        auto target_info = target.GetTensorTypeAndShapeInfo();
        if (target_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
            target_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          return Ort::GetApi().CreateStatus(
              ORT_NOT_IMPLEMENTED,
              "TargetIdCountEmbedding unsupported target dtype");
        }
        target_count = target_info.GetElementCount();
        if (target_count != 1 && target_count != rows) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "TargetIdCountEmbedding target must be scalar or per-row");
        }
        target_elem_size = ElementSize(target_info.GetElementType());
        RETURN_IF_ERROR(target_buffer.Bind(target, GetComputeStream(ctx)));
        target_data = target_buffer.data();
      }

      DeviceInputBuffer ids_buffer;
      DeviceInputBuffer table_buffer;
      RETURN_IF_ERROR(ids_buffer.Bind(ids, GetComputeStream(ctx)));
      RETURN_IF_ERROR(table_buffer.Bind(table, GetComputeStream(ctx)));
      if (ids_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        return LaunchStatus(LaunchMusaTargetIdCountEmbeddingInt32Kernel(
            static_cast<const int32_t*>(ids_buffer.data()), target_data,
            static_cast<const float*>(table_buffer.data()),
            embedding_output.GetTensorMutableData<float>(),
            count_output.GetTensorMutableData<float>(), rows, sequence_length,
            embedding_dim, table_shape[0], target_count, target_elem_size,
            values.target_constant.value_or(0), values.pad, values.cap,
            GetComputeStream(ctx)));
      }
      return LaunchStatus(LaunchMusaTargetIdCountEmbeddingInt64Kernel(
          static_cast<const int64_t*>(ids_buffer.data()), target_data,
          static_cast<const float*>(table_buffer.data()),
          embedding_output.GetTensorMutableData<float>(),
          count_output.GetTensorMutableData<float>(), rows, sequence_length,
          embedding_dim, table_shape[0], target_count, target_elem_size,
          values.target_constant.value_or(0), values.pad, values.cap,
          GetComputeStream(ctx)));
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, ex.what());
    }
  }

  TargetIdCountEmbeddingValues values;
  std::optional<size_t> target_index;
  size_t ids_index;
  size_t table_index;
  size_t embedding_output_index;
  size_t count_output_index;
};

}  // namespace

bool IsTargetIdCountEmbeddingFusionGraph(Ort::ConstGraph graph) {
  bool has_gather = false;
  bool has_reduce_sum = false;
  bool has_and = false;
  bool has_min = false;
  for (Ort::ConstNode node : graph.GetNodes()) {
    has_gather = has_gather || musa_ep::IsOnnxOp(node, "Gather");
    has_reduce_sum = has_reduce_sum || musa_ep::IsOnnxOp(node, "ReduceSum");
    has_and = has_and || musa_ep::IsOnnxOp(node, "And");
    has_min = has_min || musa_ep::IsOnnxOp(node, "Min");
  }
  TargetIdCountEmbeddingValues values;
  return has_gather && has_reduce_sum && has_and && has_min &&
         ResolveTargetIdCountEmbeddingValues(graph, values);
}

std::unique_ptr<FusionNodeCompute> CreateTargetIdCountEmbeddingFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node) {
  TargetIdCountEmbeddingValues values;
  if (!ResolveTargetIdCountEmbeddingValues(graph, values)) {
    throw std::runtime_error("unable to resolve TargetIdCountEmbedding values");
  }

  auto input_indices = ValueIndices(fused_node.GetInputs());
  auto output_indices = ValueIndices(fused_node.GetOutputs());
  std::optional<size_t> target_index;
  if (!values.target_constant.has_value()) {
    target_index = MappedIndex(input_indices, musa_ep::Name(values.target),
                               "target input");
  }
  return std::make_unique<TargetIdCountEmbeddingFusionCompute>(
      values, target_index,
      MappedIndex(input_indices, musa_ep::Name(values.ids), "ids input"),
      MappedIndex(input_indices, musa_ep::Name(values.table), "table input"),
      MappedIndex(output_indices, musa_ep::Name(values.embedding_output),
                  "embedding output"),
      MappedIndex(output_indices, musa_ep::Name(values.count_output),
                  "count output"));
}
