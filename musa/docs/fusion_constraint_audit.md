# MUSA Fusion 约束审计

日期：2026-07-07

本文审计当前 fusion matcher 和 runtime 实现中的 shape 假设、硬编码分支数、固定 rank，以及其他可能导致有价值 pattern 无法命中的限制。文中引用的优先级顺序来自生成文件 `musa/docs/fusion_priority.md`。

## 总结

大多数限制是实现约束，而不是偶然写死。现有 fusion 都是专用 runtime 路径，因此很多 rank 和 axis 检查是必要的，用来保证 fused compute 与原始 ONNX 子图等价。

初始审计中看起来过窄的主要限制如下：

1. `ConcatMatMul` 曾要求 capability 阶段具备完全静态且 equal-rank 的 shape。现在已改为比较已知维度，并在无法证明不兼容时允许符号动态维。
2. `SliceConcat` 曾要求至少 8 个 `Concat` 输入。这个启发式阈值已经移除；结构检查和 runtime shape 检查仍然保留。
3. `SplitReduce` 曾硬编码为恰好 2 个 split 输出和 2 个下游 reduce。现在 matcher 和 runtime 已支持两个或更多分支，runtime 通过成对 launch 现有 two-output MUSA kernel 实现。
4. `SplitUnsqueezeConcat` 和 `SplitConcatReorder` 要求 split 宽度相等。对当前 packed-layout kernel 来说这是合理的，但应作为实现限制记录下来。
5. 若干 fusion matcher 需要静态 inner width、静态 gamma/table 长度，或显式 initializer axes/sizes。这些限制通常是输出分配和专用 kernel launch 维度所需；但只要可行，dynamic batch 仍应继续放行。

## 当前 Fusion 约束

### 1. Split + Unsqueeze + Concat

Matcher：`FindSplitUnsqueezeConcatFusions`

当前约束：

- Pattern 是 `Split -> Unsqueeze -> Concat`，前面可选 `Reshape`，后面可选 `Transpose`。
- 至少需要两个 split 输出。
- 支持 rank-3 packed tensor 和 rank-4 stacked tensor。
- `Split` axis 必须是 2。
- `Unsqueeze` axis 必须是 0。
- `Concat` axis 必须是 0。
- 可选 `Transpose` 必须使用 `perm=[0,1,3,2]`。
- 所有 split part 必须宽度相等。
- shape 信息可以部分动态，但 rank 和 part width 必须能从 shape metadata 或 split sizes 中恢复。

评估：

- 未发现硬编码分支数；`part_count` 来自 split 输出数量。
- equal-width 和固定 axis 限制属于实现约束，对当前 packed-layout fusion 是合理的。
- rank-3/rank-4 限制符合这个专用 reorder 路径的预期。

建议：

- 保持现状，除非后续为 variable-width part 或不同 stack axis 增加新的 runtime kernel。

### 2. Split + Concat + Reorder

Matcher：`FindSplitConcatReorderFusions`

当前约束：

- Pattern 是 `Reshape -> Split -> Concat`。
- 至少需要两个 split 输出。
- reshape 之后的输入必须是 rank 3。
- `Split` axis 必须是 2。
- `Concat` axis 必须是 0。
- sequence 和 packed width 必须静态已知。
- split 宽度必须相等。
- Concat 输出 shape 必须是 `[batch * part_count, sequence, part_width]`。

评估：

- 未发现固定分支数。
- equal-width 和 rank-3 限制是实现限制，但与这个 fusion 实现的数据布局转换一致。
- 静态 sequence/width 要求比 dynamic-batch 处理更严格，但当前输出 shape 和 copy 计算大概率需要这些信息。

建议：

- 对当前 kernel 保持现状。
- 如果 dynamic sequence 或 variable-width part 成为常见 miss，先补测试，再考虑扩展 runtime shape 推导。

### 3. Concat + MatMul

Matcher：`FindConcatMatMulFusions`

当前约束：

- Pattern 是 `Concat -> MatMul`，其中 `Concat` 可以是 MatMul 的任一输入。
- 至少需要两个 concat 输入。
- `Concat` 输出只能喂给匹配到的 `MatMul`。
- 所有 concat 输入和 MatMul 的另一个输入都必须有 rank 信息。
- MatMul 输入必须 equal-rank，且 rank 至少为 2。
- concat 输入之间的已知 non-concat 维度必须兼容。
- 已知 MatMul batch 维度必须兼容。
- 已知 MatMul K 维度必须兼容。

评估：

- 未发现硬编码分支数。
- 原先的全静态 shape 要求过于保守，会拒绝 dynamic batch，即使所有 non-batch 维度都已知且兼容。该限制已经放宽。
- equal-rank 限制是真实的实现限制：ONNX MatMul 支持 rank broadcasting，而这个 fusion 只处理 equal-rank 输入。

建议：

- 保留 known-dimension matcher。
- 除非 runtime 扩展支持 MatMul broadcast，否则保留 equal-rank 限制。

### 4. Concat + Split

Matcher：`FindConcatSplitFusions`

当前约束：

- Pattern 是 `Concat -> Split`，可以把 split 输出的下游 `Concat` 或 `Sum` consumer 一起纳入 fused node。
- `Concat` 和 `Split` axis 都必须是 1。
- 输入必须是 rank 2，并且宽度为静态正数。
- Split sizes 必须是 initializer，并且必须匹配 split outputs。
- Split segment 不能跨越原始 concat 输入边界。
- 下游 `Sum` fusion 只支持宽度相等的 summed split outputs。

评估：

- 未发现固定分支数。
- rank-2/axis-1/static-width 限制属于实现约束，当前 segment-copy runtime 路径需要这些信息。
- 下游 `Sum` 的 equal-width 限制是 runtime 简化。

建议：

- 核心 matcher 保持现状。
- 记录下游 `Sum` 支持范围比通用 ONNX `Sum` 更窄。

### 5. Slice + Concat

Matcher：`FindSliceConcatFusions`

当前约束：

- Pattern 是 `Concat(axis=1)`，其输入包括 `Slice` 输出、zero `ConstantOfShape` 输出，或直接的 rank-2 float tensor。
- 至少需要一个 concat 输入。
- 对 direct 或 constant 输入，要求 rank-2 tensor 且静态正 column width。
- Slice axes 限制为 rank-2 的 axis 0/1。
- Slice steps 必须是 1。
- 对 row-axis slicing，必须等价于保留完整 row range。
- 至少需要一个可融合的 slice 或 zero constant 输入。

评估：

- 原先的 `>= 8` 输入阈值是任意启发式限制，不是该转换的语义要求。该阈值已经移除。
- rank-2/axis-1/static-width 限制属于实现约束，对当前 column-copy kernel 是合理的。

建议：

- 保留结构性 guard。如果后续担心收益，应该增加基于测量的 cost guard，而不是固定分支数阈值。

### 6. Tile + Concat

Matcher：`FindTileConcatFusions`

当前约束：

- Pattern 是一个 `Concat`，且至少一个输入由 `Tile` 产生。
- Concat 输出不能是 graph output。
- 每个被选择的 Tile 必须恰好有两个输入和一个输出。
- 每个 Tile 输出只能喂给匹配到的 Concat，并且 consumer slot 必须符合预期。
- 如果被选择节点之间存在外部路径，则拒绝 fusion。

评估：

- 未发现硬编码分支数。
- shape 约束主要推迟到 runtime；matcher 以结构检查为主。
- single-consumer 和 no-external-path 限制是安全要求。

建议：

- 保持现状。

### 7. Gemm + Activation

Matcher：`FindGemmActivationFusions`

当前约束：

- Pattern 是 `Gemm -> activation`。
- Activation 必须是受支持的 linear activation。
- Gemm 输出必须只有一个 consumer。
- Gemm 输入数量必须是 2 或 3。
- 如果静态 shape 可用，Gemm A 和 B 都必须是 rank 2。

评估：

- 无分支数问题。
- 静态 shape 使用是宽松的：缺少 shape 不会阻止 fusion。
- rank-2 Gemm 与 ONNX Gemm 语义和 linear runtime 一致。

建议：

- 保持现状。

### 8. MatMul + Add / MatMul + Add + Activation

Matcher：`FindFusedGemmFusions`

当前约束：

- Pattern 是直接的 `MatMul -> Add`，后面可选受支持的 activation。
- MatMul 输出必须只有一个 consumer。
- 不匹配 `MatMul -> Reshape -> Add`。
- 对 `MatMul + Add`，B 必须有已知静态 rank 2。
- 对 `MatMul + Add + Activation`，B 可以 unknown；但如果已知，必须是 rank 2。
- 如果 bias shape 已知，必须是 `[N]` 或 `[1, N]`。

评估：

- 无分支数问题。
- direct-chain 要求是已知限制；它有意避免宽泛删除 reshape。
- `MatMul + Add` 比 activation 变体更严格，因为它要求 B shape 静态已知。

建议：

- 保留 direct-chain guard。
- 如果 runtime validation 足够，可以考虑让 `MatMul + Add` 与 activation 变体对齐，允许 unknown B shape。

### 9. Shape + Reshape

Matcher：`FindShapeReshapeFusions`

当前约束：

- Pattern 是 shape metadata 构造，喂给一个或多个 `Reshape` 节点。
- 支持 `Shape`、`Gather`、`Cast`、`Concat`、initializer shape fragments，以及最终 `Reshape` target。
- 要求 small int initializer metadata；这些值必须在 compile/runtime 被 materialize。
- 拒绝 shape graph outputs 和外部 consumers。

评估：

- 无分支数问题。
- 静态 initializer 要求是合理的，因为这个 fusion 会替换 metadata 执行，并且必须知道要构造哪些维度。

建议：

- 保持现状。

### 10. Centered Reduce

Matcher：`FindCenteredReduceFusions`

当前约束：

- Pattern 是第一次 reduce、减均值、平方、第二次 reduce。
- Reduce op 必须是 `ReduceSum` 或 `ReduceProd`。
- 两个 reduce axes 都必须是最后一维。
- `keepdims` 必须是 1。
- 输入 rank 至少为 2。
- 最后一维必须已知且为正数。
- Runtime 还要求所有非最后一维的 runtime dimensions 都是 concrete。

评估：

- 无分支数问题。
- last-dimension 和 rank 要求是当前 reduction kernel 的固有限制。
- 要求 concrete runtime dimensions 符合 launch geometry 的预期。

建议：

- 保持现状。
- 如果 dynamic non-last dims 只在 runtime 造成 miss，可以评估 ORT 是否能通过 `KernelContext` 提供足够的 concrete runtime tensor shape，从而放宽 compile-time guard。

### 11. Split + Reduce

Matcher：`FindSplitReduceFusions`

当前约束：

- Pattern 是 `Split(axis=1)`，有两个或更多输出，每个输出喂给一个 `ReduceProd` 或 `ReduceMean`。
- Split 输入必须是 rank 3。
- Axis 维度和 inner 维度必须已知且为正数。
- Split sizes 必须是 small int initializer，并且每个 split 输出对应一个 size。
- 每个 split 输出必须恰好有一个 reduce consumer。
- Reductions 必须使用 axis 1，并且 `keepdims=0`。

评估：

- 原先恰好 2 分支的限制已经移除。
- Runtime 仍复用现有 two-output MUSA kernel，按 pair launch 分支。奇数分支数时，最后一组会使用一个临时 MUSA output 接收未使用的第二个结果。
- rank/axis/reduce-op 限制仍是当前 kernel 的实现限制。

建议：

- 保留 vectorized branch metadata 和 pairwise launch 结构。
- 只有当 profile 显示 pairwise launch 对常见模型代价过高时，再增加原生 N-output kernel。

### 12. RMS Norm

Matcher：`FindRmsNormFusions`

当前约束：

- Pattern 是 `x*x -> ReduceMean(last dim, keepdims=1) -> Add(epsilon) -> Sqrt -> Div(x, denom) -> Mul(gamma)`。
- Gamma 必须是 rank 1，且长度已知为正。
- 输入 rank 至少为 2。
- 如果输入最后一维已知，必须等于 gamma length。
- Reduce axis 必须是最后一维。
- Epsilon 必须是 scalar float initializer。
- 中间值必须只有预期 consumer。

评估：

- 无分支数问题。
- rank/gamma/last-dim 限制符合 RMSNorm 预期。
- 当前 kernel 需要静态 gamma length。

建议：

- 保持现状。

### 13. Modulo + Gather

Matcher：`FindModuloGatherFusions`

当前约束：

- Pattern 是一个特定的 reserved-bucket modulo/index rewrite，喂给 `Gather`。
- Gather axis 必须是 0。
- Modulus、offset 和 invalid sentinel 必须是 scalar int initializers。
- Gather table 第一维必须已知且为正。
- Offset plus modulus 必须落在 table rows 范围内。
- 中间值不能逃逸。

评估：

- 无分支数问题。
- 这是有意保持很窄的模型特定 pattern fusion。
- 静态 scalar/table 检查是安全所需。

建议：

- 保持现状。

### 14. Parallel MatMul + Concat

Matcher：`FindParallelMatMulConcatFusions`

当前约束：

- Pattern 是多个 `MatMul(X, Wi) -> Unsqueeze -> Concat`。
- 至少需要两个分支；没有固定上限。
- 所有 MatMul 左输入必须是同一个 graph value。
- 每个 weight 必须是 float、rank 2，并且静态 shaped。
- 所有 weights 必须 shape 相同。
- MatMul output rank 至少为 2。
- Unsqueeze 和 Concat 必须把 branch 维插入到 MatMul output 最后一维之前。
- Concat 输出不能是 graph output。

评估：

- 之前硬编码 4 分支和 input-shape 限制已经移除。
- 当前分支数行为是合理的。
- identical weight shape 是当前输出布局 `[..., branches, N]` 所需；variable N 需要不同输出布局，且与当前 Concat shape 不等价。

建议：

- 保留当前限制。
- 继续把该 fusion 放在最低 `GetCapability` 优先级，使未来更大的 fusion 可以优先 claim 相同节点。

## Action Items

已完成清理：

1. `ConcatMatMul` 已从 static-shape 检查放宽到 known-dimension 检查。
2. 已移除 `SliceConcat` 的 `concat_inputs.size() >= 8` 阈值。
3. `SplitReduce` matcher 和 runtime 已扩展为支持两个或更多分支。
4. 已为放宽后的场景增加 focused fusion tests：
   `test_dynamic_concat_matmul_equal_rank_fuses`、
   `test_slice_concat_fusion_three_segments_fuses` 和
   `test_split_reduce_three_way_fusion`。

剩余 follow-up：

1. 如果 split/reorder fusions 的 equal-width 约束继续作为有意限制保留，应在生成文档或手写 fusion 文档中记录。
