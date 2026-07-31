# MUSA Fusion 优先级重排验证报告

## 目的

本文记录 fusion 优先级重排前后的真实模型命中情况和 E2E 回归结果。目标不是只验证代码能够构建，而是确认：

- 更专用的 fusion 不会被更通用的 fusion 抢占；
- 已有有效 fusion 不会无原因消失；
- runtime graph 不会退回更多普通算子；
- 单元测试及 JD platform 模型的 MUSA E2E 数值验证均通过。

优先级设计见 [`fusion_priority_reordering_proposal.md`](fusion_priority_reordering_proposal.md)，当前代码生成的顺序见 [`fusion_priority.md`](fusion_priority.md)。

## 验收口径

不能只要求“fusion node 总数不减少”。一个更专用的 fusion 可能替代多个较小 fusion，此时 fusion node 数量会下降，但融合覆盖更完整、最终 runtime compute node 更少，仍然是正向变化。

本次重排同时比较以下指标：

1. 每种标记为 `fusion` 的 runtime compute 类型及命中数量；
2. fusion node 总数；
3. runtime compute node 总数；
4. 专用 fusion 与通用 fusion 的替代关系；
5. 原先已融合的核心链是否退回普通 runtime op；
6. CPU/MUSA 输出对分、单流/多流 E2E 和 fusion 单元测试结果。

允许的变化：

- 命中情况完全不变；
- 专用 fusion 增加，同时被其覆盖的通用 fusion 减少；
- fusion node 总数减少，但 runtime compute node 总数同步减少，且能证明较大的专用 fusion 替代了多个小 fusion。

不允许的变化：

- 专用 fusion 无解释地减少或消失；
- fusion 减少后对应链退回普通 op；
- runtime compute node 增加且没有明确的正确性原因；
- 任一数值、单元测试或多流 E2E 回归。

## 测试对象

2026-07-31 扫描 `/home/albert/Projects/onnx-model-zoo/JD/real_models/platform`，共发现 6 个 ONNX 模型：

| 简称 | 模型文件 |
| --- | --- |
| `dnn_general_scene` | `dnn__general-multi-scene-pool-u2x-v1__2025051216545407__tf_models__frozen_grpah.onnx` |
| `dnn_general_task` | `dnn__general-multi-task-v8-6__2025073013525991__tf_models__frozen_grpah.onnx` |
| `dnn_multi_domain` | `dnn__multi-domain-task-simplifyv3__2026022622335783__tf_models__frozen_grpah.onnx` |
| `dnn_xinpin` | `dnn__xinpin-multi-scene-pool-u2x-v1__2026042910553934__tf_models__frozen_grpah.onnx` |
| `sim` | `sim_graph_2_frozen_graph.onnx` |
| `twin` | `twin_graph_2_frozen_graph.onnx` |

## 重排前基线

### 环境和采集方法

- Device：`MUSA_VISIBLE_DEVICES=6`
- Python：`/home/albert/Projects/onnxruntime-musa/.venv/bin/python`
- MUSA plugin：`.venv/lib/python3.11/site-packages/onnxruntime_musa/libonnxruntime_providers_musa_plugin.so`
- Plugin SHA256：`252cb5543e13b15b635f5a8fef8ce371511e6dde6c6d70ce4d49cdbdee5d9200`
- Dump runner：`onnx_runner_multi_stream.py --ep musa -s 1 -q 1 -w 0`
- 输入：runner 为这 6 个 platform 模型生成 batch-1 random input；6 个模型均实际完成一次 MUSA inference。

每个模型在独立工作目录执行，但按要求统一使用相对 dump 路径：

```bash
MUSA_VISIBLE_DEVICES=6 \
PYTHONPATH=/home/albert/Projects/onnx-model-zoo \
ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID=1 \
ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID_PATH=./sim_rt.mmd \
/home/albert/Projects/onnxruntime-musa/.venv/bin/python \
  /home/albert/Projects/onnx-model-zoo/onnx_runner_multi_stream.py \
  --ep musa -s 1 -q 1 -w 0 <model.onnx>
```

`s=1, q=1` 只用于生成 session 级 runtime graph 和做最小设备 smoke。查询数不会改变 session 构建时已经确定的 fusion partition；完整数值和多流 E2E 结果在重排后验证章节单独记录。

原始 baseline dump 保存在：

```text
/tmp/fusion_priority_baseline/<model-short-name>/sim_rt.mmd
```

### 每个模型汇总

| 模型 | Runtime compute nodes | Fusion nodes | Fusion 种类 | Dump/推理状态 |
| --- | ---: | ---: | ---: | --- |
| `dnn_general_scene` | 594 | 43 | 8 | PASS |
| `dnn_general_task` | 594 | 43 | 8 | PASS |
| `dnn_multi_domain` | 73 | 26 | 8 | PASS |
| `dnn_xinpin` | 594 | 43 | 8 | PASS |
| `sim` | 408 | 110 | 13 | PASS |
| `twin` | 337 | 82 | 9 | PASS |
| **总计** | **2600** | **347** | **17 种去重** | **6/6 PASS** |

`Runtime compute nodes` 统计 Mermaid 中的 `exec_<id>` 定义；`Fusion nodes` 统计 label 中带 `<br/>fusion<br/>` 标记的节点。不能只按类名后缀 `*FusionCompute` 统计，因为 `SegmentMaxBroadcastCompute` 和 `TargetIdCountEmbeddingCompute` 同样是 fusion runtime node。前者是最终 runtime graph 的 compute node 数，不等同于原始 ONNX node 数。

### Fusion 类型和数量

| Fusion compute | general_scene | general_task | multi_domain | xinpin | sim | twin | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BucketizeGatherFusionCompute` | 0 | 0 | 0 | 0 | 16 | 0 | 16 |
| `ConcatMatMulFusionCompute` | 2 | 2 | 0 | 2 | 0 | 0 | 6 |
| `ConcatReshapeFusionCompute` | 3 | 3 | 0 | 3 | 6 | 7 | 22 |
| `LinearFusionCompute` | 0 | 0 | 12 | 0 | 55 | 25 | 92 |
| `MaskedEmbeddingLookupFusionCompute` | 0 | 0 | 0 | 0 | 7 | 0 | 7 |
| `MathConcatLogFusionCompute` | 1 | 1 | 1 | 1 | 2 | 2 | 8 |
| `MhtaScaledDotProductAttentionFusionCompute` | 0 | 0 | 0 | 0 | 4 | 6 | 10 |
| `ModuloGatherFusionCompute` | 19 | 19 | 4 | 19 | 0 | 16 | 77 |
| `ParallelEinsumActivationFusionCompute` | 0 | 0 | 2 | 0 | 2 | 0 | 4 |
| `ParallelLinearFusionCompute` | 12 | 12 | 2 | 12 | 7 | 9 | 54 |
| `ParallelMatMulConcatFusionCompute` | 2 | 2 | 2 | 2 | 2 | 0 | 10 |
| `ReplaceInvalidIdFusionCompute` | 0 | 0 | 0 | 0 | 2 | 3 | 5 |
| `RmsNormFusionCompute` | 1 | 1 | 1 | 1 | 2 | 2 | 8 |
| `SegmentMaxBroadcastCompute` | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| `SparseIdToMaskFusionCompute` | 0 | 0 | 0 | 0 | 4 | 0 | 4 |
| `TargetIdCountEmbeddingCompute` | 0 | 0 | 0 | 0 | 0 | 12 | 12 |
| `TileConcatFusionCompute` | 3 | 3 | 2 | 3 | 0 | 0 | 11 |
| **合计** | **43** | **43** | **26** | **43** | **110** | **82** | **347** |

### 基线观察

- 3 个 general-scene/task/xinpin 模型的最终 runtime graph 和 fusion 统计完全一致。
- `dnn_multi_domain` 主要命中 `LinearFusionCompute`，并命中两组 `ParallelEinsumActivationFusionCompute`。
- `sim` 命中种类最多，共 13 种，并包含 1 个 `SegmentMaxBroadcastCompute`；它是本次优先级重排最重要的 overlap 回归模型。
- `twin` 命中 6 个 MHTA attention、25 个 linear、16 个 modulo gather 和 12 个 `TargetIdCountEmbeddingCompute`。
- 6 个模型中没有命中 Reduced MHA Flash、Centered Reduce、Split 系列、Shape Reshape、Strided View 等 fusion；这些 fusion 的回归主要依赖 `test/fusion` 单元测试。

### 重排前 E2E 基线

在修改优先级前执行：

```bash
cd /home/albert/Projects/onnx-model-zoo
MUSA_VISIBLE_DEVICES=6 \
bash run_all_onnx_cpu_ep_diff.sh \
  --ep musa -s 4 -q 10 JD/real_models/platform
```

结果为 `pass=5 fail=1 skip=0`：4 个 DNN 模型和 TWIN 均通过，所有通过模型的最大绝对/相对误差均为 0。SIM 有 2/10 个 query 失败，观察到 `max_abs_diff=0.674999997`、`max_rel_diff=0.899999996`。这是重排前已经存在的多流 E2E 基线现象，不能把重排后的结果与一个假定的“基线 6/6”比较。

## 重排后对比

### 构建和产物确认

- 构建命令：`cmake --build build/Release --target onnxruntime_providers_musa_plugin -j8`
- 重排后 plugin SHA256：`32f22671248f35de1d1598d0c21c3a26c5abe23e1e253d5c817ee8d5e8997613`
- `build/Release` 与 `.venv/lib/python3.11/site-packages/onnxruntime_musa` 中的 plugin SHA256 一致。
- 重排后 dump：`/tmp/fusion_priority_after/<model-short-name>/sim_rt.mmd`

构建过程重新生成了 [`fusion_priority.md`](fusion_priority.md) 和 `musa/docs/fusion/*.md`。最终 GetCapability 顺序为 27 个 finder，Compile 顺序为 26 个显式 detector；`Concat MatMul` 不再使用隐式 fallback。

### Runtime graph 汇总对比

| 模型 | Runtime nodes 前 | Runtime nodes 后 | Fusion 前 | Fusion 后 | 种类前 | 种类后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dnn_general_scene` | 594 | 594 | 43 | 43 | 8 | 8 |
| `dnn_general_task` | 594 | 594 | 43 | 43 | 8 | 8 |
| `dnn_multi_domain` | 73 | 73 | 26 | 26 | 8 | 8 |
| `dnn_xinpin` | 594 | 594 | 43 | 43 | 8 | 8 |
| `sim` | 408 | 408 | 110 | 110 | 13 | 13 |
| `twin` | 337 | 337 | 82 | 82 | 9 | 9 |
| **总计** | **2600** | **2600** | **347** | **347** | **17 种去重** | **17 种去重** |

对每个模型进一步提取并排序 `runtime compute 类型 + source op 序列` 后，重排前后 6/6 均为 `IDENTICAL`。因此不仅 fusion 总数相同，上表列出的每一种 fusion 数量也完全相同，且每个 fusion 覆盖的 source op 组合没有改变。

原始 Mermaid 文件的 SHA256 不同，差异来自 session 构图时分配的内部 `value_<id>` 编号；这些编号不代表 compute topology 变化。

### 单测反馈形成的顺序约束

第一次机械重排后，`test/fusion` 暴露两个问题，并据此修正最终方案：

1. `ConcatMatMul` 必须位于 `ConcatSplit` 前。后者能够吸收下游 Concat；让 `ConcatMatMul` 先接受下游 `Concat -> MatMul` 后，`ConcatSplit` 才会缩减到其余节点。
2. `StridedView` 前移到 `SliceConcat` 前时，其 Compile detector 必须像 matcher 一样校验 `Concat(axis=2)`，否则普通 `SliceConcat(axis=1)` graph 会误分派给 Strided View runtime。

最终实现还为 `ConcatMatMul` 增加了严格的两节点显式 detector，并把未知 fused graph 改为明确报错，避免继续静默落入 Concat MatMul runtime。

## 单元测试和 E2E 结果

### 单元测试

| 命令 | 结果 |
| --- | --- |
| 两个 overlap/dispatch 定向回归用例 | `2 passed in 3.27s` |
| `MUSA_VISIBLE_DEVICES=6 ./.venv/bin/python -m pytest test/fusion -q` | `103 passed in 6.04s` |
| `MUSA_VISIBLE_DEVICES=6 ./.venv/bin/python -m pytest test/ops test/fusion test/multi_stream -q` | `841 passed in 20.31s` |

### 重排后多流 E2E

使用与基线完全相同的 `-s 4 -q 10` 命令，结果为：

| 模型组 | 结果 | failed queries | max abs diff | max rel diff |
| --- | --- | ---: | ---: | ---: |
| 4 个 DNN 模型 | 4/4 PASS | 0 | 0 | 0 |
| SIM | PASS | 0 | 0 | 0 |
| TWIN | PASS | 0 | 0 | 0 |
| **合计** | **6/6 PASS** | **0** | **0** | **0** |

SIM 从基线的 2 个失败 query 变为本次 0，但 runtime graph 完全相同，因此不能把这解释为 fusion 优先级重排带来的数值修复；本报告只据此确认重排没有产生新的 E2E 退化。

### 重排后单流 E2E

补充执行同一脚本的 `-s 1 -q 10` 模式，6/6 模型全部通过，合计 `failed_q=0`，所有模型的最大绝对/相对误差均为 0。该结果用于确认重排后的单流正确性；由于重排前未单独采集单流 CPU-EP diff，它不作为前后 A/B 统计。

## 结论

本次重排满足验收要求：

- 6 个 platform 模型的 fusion 类型、数量、source op 覆盖和 runtime compute node 数全部不变；
- 完整 MUSA Python 测试集 841/841 通过；
- 重排后单流和多流 E2E 均为 6/6 通过且误差为 0；
- attention、业务复合 pattern 已前移，两个通用 single-branch linear matcher 保持在最后；
- overlap 例外不强行服从抽象的“节点数/专用度排序”，而由现有语义和单元测试固定相对顺序。

因此，“fusion node 应不变或增加”可以作为快速报警指标，但不能作为唯一标准。更合理的标准是：专用 fusion 不退化；任何 fusion 数量下降都能由更完整 fusion 替代解释；runtime compute node 不无故增加；数值和单元测试全部通过。本次结果是其中最稳妥的一种情况——所有图级统计都完全不变。
