# MUSA Fusion 优先级重排建议

## 目标

本文基于当前代码中的两条有序 dispatch 链提出重排建议：

- `musa/ep/src/fusion/fusion_matcher.cc` 中的 GetCapability matcher 顺序；
- `musa/ep/src/fusion/fusion_node_compute.cc` 中的 Compile detector/factory 顺序。

目标是让越专用、约束越强、覆盖子图越完整的 fusion 越早匹配，让越通用、越容易与其他 pattern 共享节点的 fusion 越晚匹配。典型结果是 attention、规范化和业务复合 pattern 位于前部，单分支 linear fusion 位于最后。

本文同时记录建议及落地时由单元测试确认的相对顺序约束。当前实际顺序的机械生成结果以 [`fusion_priority.md`](fusion_priority.md) 为准，真实模型和测试结果见 [`fusion_priority_reordering_validation.md`](fusion_priority_reordering_validation.md)。

## 为什么需要重排

GetCapability 不是对每个 matcher 独立收集结果后再统一选择最优解。当前流程按 finder 顺序执行：

```text
finder 提出 candidates
  -> dispatcher 校验 candidate
  -> 接受后将全部 node id 写入 accepted_node_ids
  -> 后续 finder 不能再使用这些 node
```

因此 matcher 的先后顺序直接决定 overlap 时谁获胜。通用 fusion 如果位于前面，可能先占用一个较小子图，使后面的专用 fusion 无法命中更完整的 pattern。

Compile dispatch 是另一条优先级链。它不参与原始 ONNX node 的抢占，而是把 ORT 已生成的 fused graph 分派给具体的 `FusionNodeCompute`。如果多个 `Is*FusionGraph` 能识别同一 fused graph，更具体的 detector 同样必须位于前面。

## 排序原则

建议按以下规则排序，优先级从高到低：

1. 语义专用度：attention、normalization、特定推荐业务语义优先于通用数据搬运和 linear。
2. pattern 完整度：能覆盖完整算法或多分支子图的 fusion 优先于只覆盖其中两三个节点的 fusion。
3. 约束强度：固定 op 组合、固定拓扑、严格 shape/axis/常量约束的 fusion 优先于允许较多变体的 fusion。
4. overlap 包含关系：若 A 通常覆盖 B 的核心节点并附带更多上下游语义，A 应在 B 前面。
5. 通用兜底：单分支 `Gemm/MatMul + Add/Activation` 等 linear pattern 放在最后。

不建议只按节点数排序。节点多通常意味着更专用，但 matcher 的 shape、axis、consumer 和常量约束，以及 runtime 收益也必须一起考虑。

## 建议的 GetCapability 顺序

下表给出完整的建议顺序。“当前”来自本次重构前的 `fusion_priority.md`。

| 建议 | 当前 | Fusion | 分层 | 调整理由 |
| ---: | ---: | --- | --- | --- |
| 1 | 1 | Mhta Scaled Dot Product Attention | Attention | 完整 scaled-dot-product attention 语义，继续保持最高优先级。 |
| 2 | 2 | Reduced Mha Flash | Attention | `Unsqueeze -> Attention -> Reshape -> Gemm` 的专用 attention 路径，继续位于最前部。 |
| 3 | 18 | Parallel Einsum Activation | 算法复合 | 固定多分支 Einsum/activation/Concat 结构，比通用 Concat、activation 和 linear 更专用。 |
| 4 | 16 | Rms Norm | 规范化 | 完整 RMSNorm 数学链，应先于通用 `Mul/Add` 或 reduction 类 pattern。 |
| 5 | 14 | Centered Reduce | Reduction | 两次 reduce 与中心化平方链组成完整语义，优先于结构型 fusion。 |
| 6 | 26 | Segment Max Broadcast | 业务复合 | 约 26 个节点且 detector 校验精确 op 计数，是当前最严格的业务 pattern 之一，不应接近末尾。 |
| 7 | 23 | Target Id Count Embedding | 业务复合 | count、mask、bucket 和 embedding lookup 的完整复合语义。 |
| 8 | 22 | Masked Embedding Lookup | 业务复合 | mask、NonZero、Gather、ScatterND 等组成专用 embedding 路径。 |
| 9 | 20 | Sparse Id To Mask | 业务复合 | 完整 sparse-id mask 构造；应先于可能覆盖其局部条件/替换链的 fusion。 |
| 10 | 21 | Bucketize Gather | 业务复合 | 比 Modulo Gather 多出 threshold/bucketize 和最终 Squeeze 约束，应排在 Modulo Gather 前。 |
| 11 | 17 | Modulo Gather | 业务复合 | 固定 modulo、mask、Gather 链，专用度高于普通结构型 fusion。 |
| 12 | 25 | Replace Invalid Id | 业务复合 | 固定条件选择/替换语义，但覆盖面小于 Sparse Id To Mask，因此置于其后。 |
| 13 | 19 | Math Concat Log | 业务复合 | 固定 `Max -> Add -> Log -> Mul` 数学链，仍应先于通用结构组合。 |
| 14 | 3 | Split Unsqueeze Concat | 结构复合 | Split 每路经 Unsqueeze 后再 Concat，明显比普通 Split Concat 更具体。 |
| 15 | 15 | Split Reduce | 结构复合 | `Split -> 多路 ReduceProd/ReduceMean`；应先于可能只占用上游 `Concat -> Split` 的 Concat Split。 |
| 16 | 5 | Concat MatMul | 结构 overlap 约束 | pattern 本身虽小，但必须先占用 `ConcatSplit` 下游的 `Concat -> MatMul`，随后 `ConcatSplit` 才能安全缩减到剩余节点。 |
| 17 | 6 | Concat Split | 结构复合 | 可覆盖 `Concat -> Split` 及部分下游 Concat/Sum；必须位于 Concat MatMul 之后，避免吞掉其下游 Concat。 |
| 18 | 4 | Split Concat | 结构复合 | 通用的 Split 重排/Concat 路径，位于更具体的 Split 派生 pattern 之后。 |
| 19 | 9 | Parallel MatMul Concat | 多分支计算 | 固定多路 `MatMul -> Unsqueeze -> Concat`，应先于更宽泛的 Parallel Linear。 |
| 20 | 10 | Parallel Linear | 多分支计算 | 要求同一输入的多路 MatMul/Gemm，仍比单分支 linear 专用。 |
| 21 | 27 | Strided View | 结构复合 | 要求同一 MatMul 结果的多路 Slice 再 Concat，比普通 Slice Concat 更具体，不应放在全局最后。 |
| 22 | 13 | Shape Reshape | 结构复合 | Shape/Gather/Cast/Concat 生成 reshape shape 的专用元数据路径。 |
| 23 | 8 | Tile Concat | 结构复合 | 对 Concat 输入 producer 有 Tile 约束，专用度高于普通 Concat 邻接 pattern。 |
| 24 | 7 | Slice Concat | 结构复合 | 对 Concat 输入 producer 有 Slice/zero-segment 约束，但覆盖面比 Strided View 更宽。 |
| 25 | 24 | Concat Reshape | 通用结构 | 仅要求 `Concat -> [Unsqueeze] -> Reshape`，放在 producer 约束更强的 Concat fusion 之后。 |
| 26 | 11 | Gemm Activation | 通用 Linear | 包含专用 reshape 变体，但也接受直接 `Gemm -> Activation`，作为 linear 兜底放在倒数第二。 |
| 27 | 12 | Fused Gemm | 通用 Linear | 接受常见 `MatMul -> Add [-> Activation]`，覆盖面最通用，建议作为最后一个 GetCapability matcher。 |

建议分层后的总体结构为：

```text
Attention
  -> 算法/规范化
  -> 推荐业务复合 pattern
  -> Split/Concat/多分支结构复合
  -> 通用 Concat 邻接 pattern
  -> 单分支 Linear 兜底
```

### 需要特别保持的相对顺序

- `Mhta Scaled Dot Product Attention`、`Reduced Mha Flash` 必须位于所有 linear detector/matcher 前。
- `Parallel MatMul Concat` 必须位于 `Parallel Linear` 前，二者都必须位于单分支 linear 前。
- `Sparse Id To Mask` 必须位于 `Replace Invalid Id` 前，避免较小的条件替换 pattern 抢占复合 mask 路径。
- `Bucketize Gather` 必须位于 `Modulo Gather` 前，较长且带 threshold 的变体优先。
- `Split Unsqueeze Concat` 和 `Split Reduce` 必须位于通用 Split/Concat 组合前。
- `Concat MatMul` 必须位于 `Concat Split` 前。`Concat Split` matcher 可以吸收下游 Concat；先让 `Concat MatMul` 接受该下游 pair，`Concat Split` 才会按设计缩减到剩余节点。该顺序由 `test_concat_split_skips_downstream_concat_matmul_overlap` 固定。
- `Strided View` 必须位于 `Slice Concat` 前，同时 Compile detector 必须校验 `Concat(axis=2)`。否则普通 `Slice Concat(axis=1)` fused graph 会被错误分派给 Strided View runtime。该约束由 `test_slice_concat_fusion_three_segments_fuses` 固定。
- `Gemm Activation` 位于 `Fused Gemm` 前：前者包含更长的 reshape/activation 变体，后者是更普遍的 `MatMul + Add` 入口。

## 建议的 Compile dispatch 顺序

Compile 顺序不需要和 27 个 finder 一一对应：`Gemm Activation` 与 `Fused Gemm` 共享 `IsLinearFusionGraph/CreateLinearFusion`。重排前 `Concat MatMul` 没有 detector、依赖最终 `fallback`；本次实现为它增加了显式 detector。

建议按下面的 detector 顺序重排：

| 建议 | Detector / Factory |
| ---: | --- |
| 1 | `IsMhtaScaledDotProductAttentionFusionGraph` / `CreateMhtaScaledDotProductAttentionFusion` |
| 2 | `IsReducedMhaFlashFusionGraph` / `CreateReducedMhaFlashFusion` |
| 3 | `IsParallelEinsumActivationFusionGraph` / `CreateParallelEinsumActivationFusion` |
| 4 | `IsRmsNormFusionGraph` / `CreateRmsNormFusion` |
| 5 | `IsCenteredReduceFusionGraph` / `CreateCenteredReduceFusion` |
| 6 | `IsSegmentMaxBroadcastFusionGraph` / `CreateSegmentMaxBroadcastFusion` |
| 7 | `IsTargetIdCountEmbeddingFusionGraph` / `CreateTargetIdCountEmbeddingFusion` |
| 8 | `IsMaskedEmbeddingLookupFusionGraph` / `CreateMaskedEmbeddingLookupFusion` |
| 9 | `IsSparseIdToMaskFusionGraph` / `CreateSparseIdToMaskFusion` |
| 10 | `IsBucketizeGatherFusionGraph` / `CreateBucketizeGatherFusion` |
| 11 | `IsModuloGatherFusionGraph` / `CreateModuloGatherFusion` |
| 12 | `IsReplaceInvalidIdFusionGraph` / `CreateReplaceInvalidIdFusion` |
| 13 | `IsMathConcatLogFusionGraph` / `CreateMathConcatLogFusion` |
| 14 | `IsSplitUnsqueezeConcatFusionGraph` / `CreateSplitUnsqueezeConcatFusion` |
| 15 | `IsSplitReduceFusionGraph` / `CreateSplitReduceFusion` |
| 16 | `IsConcatMatMulFusionGraph` / `CreateConcatMatMulFusion` |
| 17 | `IsConcatSplitFusionGraph` / `CreateConcatSplitFusion` |
| 18 | `IsSplitConcatFusionGraph` / `CreateSplitConcatFusion` |
| 19 | `IsParallelMatMulConcatFusionGraph` / `CreateParallelMatMulConcatFusion` |
| 20 | `IsParallelLinearFusionGraph` / `CreateParallelLinearFusion` |
| 21 | `IsStridedViewFusionGraph` / `CreateStridedViewFusion` |
| 22 | `IsShapeReshapeFusionGraph` / `CreateShapeReshapeFusion` |
| 23 | `IsTileConcatFusionGraph` / `CreateTileConcatFusion` |
| 24 | `IsSliceConcatFusionGraph` / `CreateSliceConcatFusion` |
| 25 | `IsConcatReshapeFusionGraph` / `CreateConcatReshapeFusion` |
| 26 | `IsLinearFusionGraph` / `CreateLinearFusion` |

最后使用“未知 fused graph 报错”的分支，不再把所有未识别 graph 隐式交给 `CreateConcatMatMulFusion`。显式 detector 有三个好处：

- Compile 顺序能够完整生成到 `fusion_priority.md`，不再把 Concat MatMul 标成语义含糊的 `fallback`；
- 新增 detector 时不会把漏注册或错误 matcher 的 graph 静默误分派为 Concat MatMul；
- 可以对 detector 的互斥性编写直接单元测试。

## 本次重排不应顺带修改的内容

- 不改变任何 GetCapability matcher 的 pattern、shape、axis、dtype、initializer 或 consumer 约束。
- 不改变 `drop_constant_initializers` 的现有取值；它是 runtime 输入契约，不是优先级参数。
- 不改变 kernel 实现和 runtime compute 路径。
- 不通过删除 overlap 校验来提高命中数；`accepted_node_ids` 的原子接受规则必须保留。

唯一附带的识别收紧发生在 Compile detector：`IsStridedViewFusionGraph` 增加与 matcher 一致的 `Concat(axis=2)` 校验，防止优先级前移后误识别 `SliceConcat` graph。这不是扩大 fusion capability，也不改变 GetCapability 命中集合。

## 实施建议

### 第一阶段：建立重排基线

1. 保存当前 `fusion_priority.md`、fusion 测试结果和代表性模型的 runtime graph。
2. 对已知 overlap 家族补充命中归属测试，至少覆盖：
   - attention 与 linear；
   - Parallel MatMul Concat、Parallel Linear 与单分支 linear；
   - Concat Split、Split Concat、Split Reduce；
   - Sparse Id To Mask 与 Replace Invalid Id；
   - Bucketize Gather 与 Modulo Gather；
   - Strided View 与 Slice Concat。
3. 测试不仅检查数值，还要检查最终 fused compute 类型或 runtime graph 节点名。

### 第二阶段：只重排 GetCapability

按本文表格调整 `FindFusionMatches()`，不同时修改 matcher 条件。重新生成 `fusion_priority.md`，确认 27 个 finder 的顺序和 `drop_constant_initializers` 均符合预期。

### 第三阶段：整理 Compile dispatch

1. 按具体程度重排 detector。
2. 为 Concat MatMul 增加显式 detector。
3. 将最终 fallback 改为清晰的未知 fusion 错误。
4. 为可能共享 op 集合的 detector 增加互斥性测试，避免仅依赖 `else if` 顺序掩盖 detector 过宽。

长期看，可以把 finder、`drop_constant_initializers`、detector、factory 和分层信息收敛到同一份注册表，再由代码生成 GetCapability/Compile dispatch 与文档，避免两条手写顺序逐渐漂移。本次重排不必依赖这项结构改造，可以先完成纯顺序变更。

## 验收标准

完成重排后，至少满足以下条件：

1. `scripts/gen_fusion_docs.py` 生成的 `fusion_priority.md` 与本文建议顺序一致。
2. `test/fusion` 全量通过，已有 fusion 的 CPU/MUSA 数值对比不退化。
3. overlap 测试证明专用 fusion 获胜，且一个原始 ONNX node 最多属于一个 accepted fusion。
4. 代表性 attention、推荐、Split/Concat 和 linear 模型的 runtime graph 中，fusion 类型与预期一致。
5. 对比重排前后：专用 fusion 命中数不下降；若通用 fusion 命中数下降，能够由更高优先级、更完整的 fusion 命中解释。
6. 多流和单流真实模型回归均通过，避免顺序变化只在最小测试中正确。

## 后续新增 fusion 的规则

“新增 fusion 默认放最低优先级”仍然是安全默认值，但不应把它永久留在末尾。建议采用两步策略：

1. 首次合入时追加到末尾，降低对现有模型的影响；
2. 有 overlap 测试和真实模型证据后，按本文分层规则把它提升到对应 family，并在 review 中明确它必须先于哪些通用 fusion。

这样既保留新增功能时的保守策略，也能避免长期积累成“越晚开发、优先级越低”的历史顺序。
