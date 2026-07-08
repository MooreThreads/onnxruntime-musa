# MUSA Fusion Development Guide

本文档说明当前 MUSA EP fusion 的开发范式。它是手写开发指南；自动生成的优先级和单个 fusion 拓扑说明仍然在 [`fusion_priority.md`](fusion_priority.md) 和 [`fusion/`](fusion/) 下。关于 overlap 机制迁移的背景和修复细节，见 [`fusion_overlap_fix.md`](fusion_overlap_fix.md)。

## 当前模型

MUSA fusion 不是 ORT `GraphTransformer`，也不是把 ONNX 子图替换成另一个 ONNX 子图。当前路径是：

```text
GetCapability matcher
  -> EpGraphSupportInfo_AddNodesToFuse
  -> ORT 生成 fused node
  -> CompileImpl dispatch
  -> FusionNodeCompute::Compute
```

也就是说：

- GetCapability 阶段只负责识别可以安全融合的 ONNX 子图。
- Compile 阶段负责把 ORT 生成的 fused node 转成具体 `FusionNodeCompute`。
- Runtime 阶段只执行 fused compute，不再重新做复杂图匹配。

## 设计原则

新增 fusion 前先判断能否扩展已有 fusion。如果只是给已有 pattern 增加一个小变体，应优先修改已有 matcher/runtime，而不是新增一套并列 fusion。例如 `MatMul + Add + Activation` 中新增一种 activation 支持，应优先扩展现有 linear fusion 的 activation 判定和 runtime 分支。只有当新 pattern 的节点结构、输入输出约定、runtime compute 或安全约束已经明显不同，才新建独立 fusion。

新建 fusion 的默认 GetCapability 优先级应放在最低，也就是追加到 `fusion_matcher.cc` 的现有 matcher 顺序末尾。这样不会抢占已有 matcher 的节点，也不会改变其他模型和开发者已经依赖的融合行为。确实需要插到更高优先级时，必须在代码 review 或相关文档中说明原因，并证明它不会破坏已有 pattern 的预期命中。

## 文件职责

新增或修改 fusion 时，优先按下面边界放代码：

| 文件 | 职责 |
| --- | --- |
| `musa/ep/src/fusion/fusion_matcher.cc` | GetCapability 侧 matcher 调度和优先级顺序 |
| `musa/ep/src/fusion/fusion_matcher.h` | matcher 对外声明，供调度文件和 `ep.cc` 使用 |
| `musa/ep/src/fusion/fusion_matcher_utils.{h,cc}` | 跨 matcher 复用的 producer/consumer/path helper |
| `musa/ep/src/fusion/<name>_fusion_matcher.cc` | 单个 fusion 的 `CanFuse*` / `Find*Fusions` |
| `musa/ep/src/fusion/<name>_fusion.{h,cc}` | Compile/runtime 侧 detector、factory 和 `FusionNodeCompute` |
| `musa/ep/src/fusion/fusion_node_compute.cc` | Compile 阶段 dispatch 到具体 `Create*Fusion` |
| `test/fusion/` | fusion 级别测试和文档生成可提取的 pattern case |

`ep.cc` 不应该再承载具体 matcher 细节。它只消费 `FindFusionMatches(...)` 的结果，并注册 fused nodes 和普通单节点 capability。

## 开发步骤

1. 先判断是扩展已有 fusion 还是新建 fusion
   - 如果已有 fusion 做小改动即可覆盖新 case，优先修改已有 fusion。
   - 如果要新建 fusion，先明确为什么不能复用或扩展已有 fusion。
   - 如果要把新建 fusion 放到非最低优先级，先明确它必须抢在谁前面以及原因。

2. 定义 pattern 的边界
   - 明确要融合的 ONNX op 序列、输入输出、shape/rank/dtype 限制。
   - 明确哪些中间 tensor 不能是 graph output。
   - 明确每个内部 tensor 是否必须 single-consumer。
   - 明确是否允许常量 initializer 作为 runtime input。

3. 写 GetCapability matcher
   - 新增 `musa/ep/src/fusion/<name>_fusion_matcher.cc`。
   - 实现 pattern-local `CanFuse*` helper。
   - 实现 `Find<Name>Fusions(...)`：

```cpp
std::vector<std::vector<Ort::ConstNode>> FindNameFusions(
    const std::vector<Ort::ConstNode>& all_nodes,
    const std::unordered_set<std::string>& graph_output_names,
    const std::unordered_set<size_t>& accepted_node_ids);
```

   - `accepted_node_ids` 是只读剪枝提示，表示更高优先级 matcher 已经被
     dispatcher 接受的原始 ONNX node。matcher 可以用它跳过 root、producer
     或 consumer，但不能修改它。
   - `Find<Name>Fusions(...)` 返回的是 candidate proposals，不代表这些
     candidate 已经最终注册给 ORT。

4. 声明 matcher
   - 在 `fusion_matcher.h` 声明 `Find<Name>Fusions(...)`。
   - 如果确实需要跨 matcher guard，优先暴露一个很小的 predicate，而不是让两个 matcher 文件互相共享大块实现。

5. 注册 matcher 优先级
   - 在 `fusion_matcher.cc` 的 `FindFusionMatches(...)` 中按优先级调用 matcher。
   - 新建 fusion 默认追加到当前 matcher 顺序末尾，除非有明确理由需要更高优先级。
   - `accepted_node_ids` 是优先级机制的一部分：前面的 matcher candidate
     被 dispatcher 接受后，后面的 matcher 不能复用这些节点。
   - 不要在 matcher 内更新 `accepted_node_ids`。全局占用只能由
     `fusion_matcher.cc` 的 dispatcher accept helper 完成。
   - `AddFusionMatch(..., drop_constant_initializers, ...)` 的第三个参数必须按 fusion runtime 输入约定设置。

6. 写 runtime fusion
   - 新增或更新 `musa/ep/src/fusion/<name>_fusion.h` 和 `.cc`。
   - 提供 detector：

```cpp
bool IsNameFusionGraph(Ort::ConstGraph graph);
```

   - 提供 factory：

```cpp
std::unique_ptr<FusionNodeCompute> CreateNameFusion(
    Ort::ConstGraph graph, Ort::ConstNode fused_node);
```

   - 实现 `FusionNodeCompute::Compute(...)`，实际计算必须在 MUSA device 路径上完成。不要为了“支持 fusion”把主计算偷偷搬到 CPU。

7. 注册 Compile dispatch
   - 在 `fusion_node_compute.cc` 中包含 `<name>_fusion.h`。
   - 在 `MusaEp::CompileImpl(...)` 的 dispatch 链中添加：

```cpp
} else if (IsNameFusionGraph(graph)) {
  fusion_compute = CreateNameFusion(graph, fused_node);
```

   - 顺序通常按 detector 的具体程度放置。多个 runtime factory 能识别同一 fused graph 时，要把更具体的 detector 放在前面。

8. 添加测试和文档生成支持
   - 在 `test/fusion/` 添加最小 ONNX pattern 测试。
   - 测试应证明 pattern 被 MUSA EP 融合，并和 CPU EP 输出一致。
   - 如果 pattern 有重要拒绝条件，也要补负例或覆盖真实触发模型。
   - 运行 `scripts/gen_fusion_docs.py` 或通过构建触发 POST_BUILD，确认 `fusion_priority.md` 和 `fusion/<name>.md` 正确生成。

## Matcher 规则

matcher 必须保守。宁可不融合，也不要返回一个 runtime factory 处理不了的子图。

基本要求：

- 每个候选节点先检查 `IsOnnxOp(...)`。
- 对 shape/rank/dtype/axis/attribute 做 capability-time 检查。
- 对内部 tensor 检查 graph output 和 consumer。
- 对已接受节点检查 `accepted_node_ids`。
- 命中后只返回 candidate，不要把 fusion 内节点 id 加入
  `accepted_node_ids`。
- 返回的 `std::vector<Ort::ConstNode>` 应包含 runtime factory 需要看到的完整子图节点。
- 返回的 candidate 内部不能包含重复 node。
- matcher 条件必须和 `Create*Fusion(...)` / `Is*FusionGraph(...)` 的约束一致。

### Candidate 和 overlap 规则

GetCapability matcher 的职责是提出候选 fusion；是否最终接受由
`fusion_matcher.cc` 中的 dispatcher 决定。dispatcher 会对每个 candidate
做统一校验：

```text
1. candidate 不能为空。
2. candidate 不能包含 null node。
3. candidate 内部不能有重复 node id。
4. candidate 不能包含已经被更高优先级 fusion 接受的 node id。
```

通过校验后，dispatcher 才会一次性把 candidate 内所有 node id 加入全局
`accepted_node_ids`。如果 candidate 和已接受 fusion 重叠，整个 candidate
会被拒绝，不会部分修改全局占用集合。

这条规则的目的是保证一个原始 ONNX node 最多只能属于一个 fused subgraph。
开发 matcher 时不要依赖“自己 insert 到全局 set”来解决冲突；这个旧模式
已经被移除。

复杂 matcher 在沿 producer 或 consumer 扩展时仍然应该主动读取
`accepted_node_ids` 做剪枝，尤其是可选 downstream node。例如某个 fusion
可以融合 `Root + OptionalConsumer`，但 `OptionalConsumer` 已被更高优先级
fusion 接受，那么 matcher 应该优先返回缩小后的 `Root` candidate，而不是
把已接受 node 放进 candidate 后等待 dispatcher 拒绝整个 fusion。

如果某个 fusion 不能安全缩小，则可以返回完整 candidate；dispatcher 会在
overlap 时拒绝它。这个行为比 matcher 自行占用 node 更安全，因为全局
no-overlap 不变量仍由 dispatcher 保证。

常用 helper：

| Helper | 位置 | 用途 |
| --- | --- | --- |
| `IsOnnxOp`、`GetTensorShape`、`GetStaticShape`、`GetIntAttribute` | `graph/graph_utils.h` | 基础 op/shape/attribute 读取 |
| `HasSingleConsumerAt`、`GetProducer`、`HasOnlyConsumer` | `fusion_matcher_utils.h` | producer/consumer 检查 |
| `AddFusionNode`、`FusionHasNoExternalPathBetweenSelectedNodes` | `fusion_matcher_utils.h` | 多节点 pattern 的安全收集和外部路径检查 |
| `ReadScalarIntInitializer`、`IsSmallIntegerInitializer` | `fusion_matcher_utils.h` | 小常量 initializer 检查 |

## `drop_constant_initializers`

`drop_constant_initializers` 控制 ORT 生成 fused node 时是否把常量 initializer 继续暴露成 runtime input。

- `false`：常量仍可能作为 fused node input 出现，runtime factory/compute 需要按 fused graph 输入处理。
- `true`：适合 runtime fusion 自己读取并持有常量信息，不希望 ORT 再把这些 initializer 当成 runtime input 暴露。

这个值在 `fusion_matcher.cc` 的 `AddFusionMatch(...)` 调用里设置。不要只因为 matcher 读了 initializer 就设成 `true`；要看 runtime factory 和 compute 的输入约定。

## Runtime fusion 规则

runtime 侧要把 matcher 的 capability 条件重新落到可执行实现上：

- `Is*FusionGraph(...)` 只识别自己能创建的 fused graph。
- `Create*Fusion(...)` 解析 fused graph 输入输出、source nodes、attributes 和常量。
- `Create*Fusion(...)` 遇到 matcher 不该放进来的形态时可以抛出清晰错误，但更好的做法是让 matcher 提前拒绝。
- `FusionNodeCompute::Compute(...)` 只做运行时 shape 和 device compute，不做大规模 pattern 搜索。
- 不要新增自动 CPU fallback。确实不支持的 dtype/rank/attribute 应在 matcher 或 factory 明确拒绝。

## 优先级

GetCapability 优先级只在 `fusion_matcher.cc` 中维护。顺序越靠前，越先被
调用，也越先获得接受 candidate 的机会。

当前机制分两步：

```text
for finder in priority order:
  candidates = FindXxxFusions(all_nodes, graph_output_names, accepted_node_ids)
  accepted = dispatcher validates candidates against accepted_node_ids
  dispatcher records accepted candidate node ids into accepted_node_ids
```

也就是说，finder 顺序决定冲突时谁获胜。较早 finder 的 candidate 一旦被
接受，后续 finder 返回的任何包含相同 node id 的 candidate 都会被拒绝。
后续 finder 可以读取 `accepted_node_ids`，提前避开这些 node，或者返回一个
不重叠的缩小 candidate。

默认情况下，新建 fusion 应放在最低优先级。只有在新 matcher 必须先于某个已有 matcher 才能得到正确或显著更优的行为时，才把它插到中间；这种情况需要在 review 说明中明确写出被影响的已有 matcher、原因和验证结果。

优先级调整前要检查：

- 新 matcher 是否会抢走旧 matcher 的节点。
- 被抢走后 runtime factory 是否等价或更严格。
- `accepted_node_ids` 是否会让后续 matcher 少命中真实模型中更重要的 fusion。
- 新 matcher 是否可能沿 producer/consumer 扩展到已被接受的 node。
- 如果发生 overlap，应该缩小 candidate 还是让 dispatcher 拒绝整个 candidate。
- `fusion_priority.md` 是否反映了预期顺序。

Compile dispatch 顺序在 `fusion_node_compute.cc` 中维护。它决定 ORT fused graph 到 runtime factory 的选择，和 GetCapability matcher 顺序是两件事。通常 GetCapability 顺序处理“哪个 pattern 先接受并占用原始 ONNX 节点”，Compile 顺序处理“这个 fused graph 用哪个 factory 创建 compute”。

## 文档和验证

新增或修改 fusion 后建议至少运行：

```bash
cmake --build build/Release --target onnxruntime_providers_musa_plugin -j$(nproc)
python3 scripts/gen_fusion_docs.py
MUSA_VISIBLE_DEVICES=1 ./.venv/bin/python -m pytest test/fusion -q
```

如果修改会影响共享 matcher helper、优先级或真实模型中的 pattern，还应运行：

```bash
MUSA_VISIBLE_DEVICES=1 ./.venv/bin/python -m pytest test/ops test/fusion test/multi_stream -q
```

真实模型回归使用 sibling repo 中的命令：

```bash
cd ../onnx-model-zoo
MUSA_VISIBLE_DEVICES=1 bash run_all_onnx_cpu_ep_diff.sh --ep musa -s 4 -q 128 JD/real_models/
```

完成后检查：

- `musa/docs/fusion_priority.md` 的 GetCapability 顺序是否符合预期。
- 对应 `musa/docs/fusion/<name>.md` 是否显示正确的 `Finder implementation`、runtime factory 和 runtime implementation。
- build 输出的 plugin 是否已同步到 `.venv`，避免测试加载旧 `.so`。
