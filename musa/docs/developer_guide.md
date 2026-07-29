# MUSA EP Developer Guide

本文档整理本仓库当前用到的环境变量和相关构建变量。默认按源码实际读取路径说明；如果变量只是在脚本或文档命令中使用，会单独标注。

## 构建和安装相关

### `build.sh` 构建模式

- 默认行为：重新运行 CMake 配置后复用 `build/<Config>/`，只编译变更的源码及其依赖；默认仍构建 wheel。
- `--clean`：删除 `build/<Config>/` 和 `dist/`，然后从零构建。适用于首次构建、工具链/MUSA/ABI 变化、切换生成器或发布前的干净验证。
- `--no-wheel`：跳过 wheel 打包；与默认增量模式组合时适合日常只验证插件 `.so`。
- 示例：

```bash
./build.sh
./build.sh --no-wheel
./build.sh --clean
./build.sh --clean --no-wheel
```

### `PYTHON`

- 读取位置：`build.sh`
- 用途：指定构建 wheel 时使用的 Python 解释器。
- 默认行为：`build.sh` 优先使用显式设置的 `PYTHON`，否则依次尝试 `./.venv/bin/python`、`python3.12`、`python3.11`、`python3`。
- 示例：

```bash
PYTHON=/path/to/python ./build.sh
```

### `PATH`

- 读取位置：shell 工具链本身；`scripts/format.sh` 依赖 `command -v clang-format`。
- 用途：查找 `clang-format`、`cmake`、编译器、Python 等外部命令。
- 注意：`scripts/format.sh` 只检查 `PATH` 中是否能找到 `clang-format`。

### `LD_LIBRARY_PATH`

- 读取位置：动态链接器，不是 MUSA EP 源码直接读取。
- 用途：运行时让 Python/ONNX Runtime 能找到 MUSA toolkit 动态库，例如 `musart`、`mublas`、`mudnn` 等。
- 常用值：

```bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64:${LD_LIBRARY_PATH}
```

### `MUSA_HOME`

- 读取位置：`CMakeLists.txt` / `musa/ep/CMakeLists.txt`
- 用途：指定 MUSA toolkit 安装目录，用于 include、link directory 和 CMake module path。
- 注意：这是 CMake cache 变量，不是 C++ 运行时通过 `getenv()` 读取的环境变量。推荐通过 `-D` 传入。
- 默认值：`/usr/local/musa`
- 示例：

```bash
./build.sh -- -DMUSA_HOME=/opt/musa
cmake -S . -B build/Release -DMUSA_HOME=/opt/musa
```

## 设备选择和运行环境

### `MUSA_VISIBLE_DEVICES`

- 读取位置：MUSA runtime/toolkit，不是本仓库源码直接读取。
- 用途：限制进程可见的 MUSA 设备，常用于测试或复现指定设备上的问题。
- 示例：

```bash
MUSA_VISIBLE_DEVICES=0 ./.venv/bin/python -m pytest test/ops/test_matmul.py -q
```

### Host Metadata Inputs

Some kernels read small CPU-visible inputs such as axes, split sizes, `TopK`'s
scalar `K`, or shape tensors from host memory. The code typically marks these
inputs with `OrtMemTypeCPUInput` or copies them with `CopyToHost(...)` before
building a device launch or a shape output.

That is expected behavior for ORT metadata inputs, not a general CPU fallback.
If an operator's main tensor compute is unsupported, the kernel should reject it
explicitly instead of silently routing the whole op through the host.

## MUSA EP 调试开关

### `user_compute_stream`

- 读取位置：`musa/ep/src/ep_factory.cc`
- 用途：把调用方创建的 `musaStream_t` 作为 MUSA EP compute stream。
- 入口：通过 ORT Plugin EP V2 provider options 传入，Python 可使用 `onnxruntime_musa.make_provider_options(user_compute_stream=stream)` 生成配置。
- 表示方式：Plugin EP V2 的 provider options 是 key/value 字符串，因此 `user_compute_stream` 使用 `uintptr_t` 十进制或 `0x` 十六进制字符串；helper 会自动转换 `ctypes.c_void_p` 或整数地址。
- 行为：设置非空 `user_compute_stream` 时，EP 自动设置 `has_user_compute_stream=1` 和 `use_ep_level_unified_stream=1`。MUSA EP 包装该 stream 但不销毁，stream 生命周期由调用方管理。
- 约束：调用方必须保证 stream 属于 `device_id` 对应的 MUSA device，并且 session 仍在使用时不能销毁该 stream。

### `ORT_MUSA_DISABLE_ALL_FUSIONS`

- 读取位置：`musa/ep/src/ep.cc`
- 用途：在 `MusaEp::GetCapabilityImpl()` 中跳过全部 MUSA EP fusion matcher 和 fused node 注册，只保留普通单节点 capability。
- 默认值：未设置时关闭该禁用开关，也就是默认启用 fusion。
- 启用方式：设置为非空且不是 `0` 的值，例如 `1`。
- 关闭方式：未设置、空值或设置为 `0`。
- 注意：该开关只影响 fusion 注册，不会关闭普通 MUSA kernel。默认未设置时只在进程内首次进入该路径读取一次环境变量，不会给默认 fusion 路径增加 per-matcher 或 runtime compute 开销。
- 示例：

```bash
ORT_MUSA_DISABLE_ALL_FUSIONS=1 ./.venv/bin/python your_script.py
```

### `ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID`

- 读取位置：`musa/ep/src/graph_mermaid_dump.cc`
- 用途：在 `MusaEp::GetCapabilityImpl()` 入口处，将 ORT 传入 MUSA EP 之前的 graph dump 成 Mermaid `.mmd`。
- 默认值：未设置时关闭。
- 关闭值：空值、`0`、`false`、`off`、`no`。
- 输出内容：只输出 ONNX 算子节点和节点间依赖边；节点文本使用 op type。
- 示例：

```bash
ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID=1 ./.venv/bin/python your_script.py
```

### `ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID_PATH`

- 读取位置：`musa/ep/src/graph_mermaid_dump.cc`
- 用途：指定 GetCapability Mermaid dump 文件路径。
- 默认值：`musa_ep_get_capability_graph_<n>.mmd`
- 编号规则：
  - 路径包含 `{}` 时，用 dump 序号替换 `{}`。
  - 路径不包含 `{}` 时，第 0 次使用原路径，后续自动在扩展名前追加 `_1`、`_2` 等。
- 示例：

```bash
ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID=1 \
ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID_PATH=/tmp/musa_get_capability_graph_{}.mmd \
./.venv/bin/python your_script.py
```

### `ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID`

- 读取位置：`musa/ep/src/runtime_graph_dump.cc`
- 用途：记录 session 最终实际交给 MUSA EP 执行的 execution graph，并在进程正常退出时输出 Mermaid `.mmd`。
- 默认值：未设置时关闭。
- 关闭值：空值、`0`、`false`、`off`、`no`。
- 输出内容：普通 MUSA kernel 和 MUSA EP fused node 的最终拓扑，边由 tensor value name 建立；fusion 节点会显示实际 dispatch 到的 `*FusionCompute`，例如 `TileConcatFusionCompute`、`LinearFusionCompute`。
- 注意：这是 MUSA EP execution graph，不是 MUSA profiler 时间线；muDNN/muBLAS 内部展开出的硬件 kernel 不会在这里继续拆分。
- 示例：

```bash
ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID=1 \
ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID_PATH=/tmp/musa_runtime_execution_graph.mmd \
./.venv/bin/python your_script.py
```

### `ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID_PATH`

- 读取位置：`musa/ep/src/runtime_graph_dump.cc`
- 用途：指定 runtime execution graph Mermaid dump 文件路径。
- 默认值：`musa_ep_runtime_execution_graph.mmd`

### `MUSA_EP_TRACE_KERNELS`

- 读取位置：
  - `musa/ep/src/kernels/utils.h`
  - `musa/ep/src/kernels/shared_inc/op_kernel_common.h`
- 用途：向 `stderr` 打印 kernel 创建、kernel 开始/结束，以及部分 binary op 的形状和内存位置调试信息。
- 默认值：未设置时关闭。
- 启用方式：源码只判断是否设置，不解析具体值；设置为任意非空值都会启用，包括 `0`。
- 输出示例类别：
  - `MUSA_KERNEL_CREATE`
  - `MUSA_KERNEL_CREATED`
  - `MUSA_KERNEL_BEGIN`
  - `MUSA_KERNEL_END`
  - `MUSA_BINARY`
- 注意：该变量会增加 stderr 输出，可能影响性能测量。

### `MUSA_EP_TRACE_SYNC`

- 读取位置：`musa/ep/src/kernels/shared_inc/op_kernel_common.h`
- 用途：仅在 `MUSA_EP_TRACE_KERNELS` 已启用时生效。kernel 返回成功后额外执行一次 `musaStreamSynchronize()`，用于把异步 kernel 错误尽早暴露在对应 kernel 的 trace 位置。
- 默认值：未设置时关闭。
- 启用方式：源码只判断是否设置，不解析具体值；设置为任意非空值都会启用，包括 `0`。
- 注意：会强制同步 stream，明显改变性能和时序。只建议用于定位错误，不建议用于 benchmark。

## 内存和数据搬运调优

### `ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB`

- 读取位置：`musa/ep/src/ep_allocator.h`
- 用途：控制 MUSA device allocator 的缓存上限。
- 默认值：未设置时为 `0`，即不启用该 allocator cache。
- 解析规则：按十进制 MB 解析；小于等于 0 时视为 0。
- 示例：

```bash
ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB=512 ./.venv/bin/python your_script.py
```

### `ORT_MUSA_PINNED_POOL_CACHE_LIMIT_MB`

- 读取位置：`musa/ep/src/pinned_host_pool.h`
- 用途：控制 pinned host staging pool 的缓存上限。
- 默认值：未设置时为 `1024` MB。
- 解析规则：按十进制 MB 解析；小于等于 0 时禁用缓存，相关 block 在事件完成后释放。
- 示例：

```bash
ORT_MUSA_PINNED_POOL_CACHE_LIMIT_MB=256 ./.venv/bin/python your_script.py
```

### `ORT_MUSA_ENABLE_PAGEABLE_H2D_BOUNCE`

- 读取位置：`musa/ep/src/ep_data_transfer.cc`
- 用途：控制 pageable host 到 device 的 H2D copy 是否允许走 pinned staging bounce 路径。
- 默认值：开启。
- 关闭值：`0`、`false`、`FALSE`、`off`、`OFF`。
- 注意：只有满足其它条件时才会走 bounce 路径，例如存在 stream、pinned pool 可用、tensor 数量达到阈值、拷贝大小达到阈值等。

### `ORT_MUSA_PAGEABLE_H2D_BOUNCE_THRESHOLD_BYTES`

- 读取位置：`musa/ep/src/ep_data_transfer.cc`
- 用途：控制 pageable H2D copy 进入 pinned staging bounce 路径的最小字节数。
- 默认值：`1024` bytes。
- 行为：当单次拷贝 `bytes < threshold` 时，直接走同步 `musaMemcpy`，不走 bounce 路径。
- 解析规则：按十进制整数解析；解析失败时使用默认值。

### `ORT_MUSA_PAGEABLE_H2D_BOUNCE_MIN_TENSORS`

- 读取位置：`musa/ep/src/ep_data_transfer.cc`
- 用途：控制一次 `CopyTensors` 中至少多少个 tensor 时才允许 pageable H2D bounce。
- 默认值：`1024`。
- 行为：`num_tensors >= ORT_MUSA_PAGEABLE_H2D_BOUNCE_MIN_TENSORS` 时才允许 bounce。
- 解析规则：按十进制整数解析；解析失败时使用默认值。

## 数值行为

### `MUSA_ENABLE_TF32`

- 读取位置：`musa/ep/src/kernels/shared_inc/blas_utils.h`
- 用途：控制 muDNN handle 是否允许 TF32。
- 默认值：未设置时关闭。
- 启用方式：`std::atoi(value) != 0` 时启用，例如 `1`。
- 关闭方式：未设置或设置为 `0`。
- 示例：

```bash
MUSA_ENABLE_TF32=1 ./.venv/bin/python your_script.py
```

## 变量速查

| 变量 | 类型 | 默认值 | 主要用途 |
|---|---|---:|---|
| `PYTHON` | 构建脚本 | 自动选择 | 指定 `build.sh` 构建 wheel 使用的 Python |
| `PATH` | Shell 环境 | 系统值 | 查找外部命令，例如 `clang-format` |
| `LD_LIBRARY_PATH` | 动态链接 | 系统值 | 让运行时找到 MUSA toolkit 动态库 |
| `MUSA_HOME` | CMake cache | `/usr/local/musa` | 指定 MUSA toolkit 安装目录 |
| `MUSA_VISIBLE_DEVICES` | MUSA runtime | runtime 默认 | 限制可见 MUSA 设备 |
| `ORT_MUSA_DISABLE_ALL_FUSIONS` | EP 调试 | 关闭 | 跳过全部 MUSA EP fusion matcher 和 fused node 注册 |
| `ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID` | EP 调试 | 关闭 | GetCapability 前 dump Mermaid graph |
| `ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID_PATH` | EP 调试 | `musa_ep_get_capability_graph_<n>.mmd` | 指定 GetCapability Mermaid 输出文件 |
| `ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID` | EP 调试 | 关闭 | 运行结束时 dump runtime execution graph |
| `ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID_PATH` | EP 调试 | `musa_ep_runtime_execution_graph.mmd` | 指定 runtime execution graph Mermaid 输出文件 |
| `MUSA_EP_TRACE_KERNELS` | EP 调试 | 关闭 | 打印 kernel 创建/执行 trace |
| `MUSA_EP_TRACE_SYNC` | EP 调试 | 关闭 | trace 模式下强制 stream 同步 |
| `ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB` | 内存调优 | `0` MB | Device allocator cache 上限 |
| `ORT_MUSA_PINNED_POOL_CACHE_LIMIT_MB` | 内存调优 | `1024` MB | Pinned host pool cache 上限 |
| `ORT_MUSA_ENABLE_PAGEABLE_H2D_BOUNCE` | 数据搬运 | 开启 | 控制 pageable H2D pinned staging bounce |
| `ORT_MUSA_PAGEABLE_H2D_BOUNCE_THRESHOLD_BYTES` | 数据搬运 | `1024` bytes | 单次 H2D bounce 最小字节数 |
| `ORT_MUSA_PAGEABLE_H2D_BOUNCE_MIN_TENSORS` | 数据搬运 | `1024` | CopyTensors 允许 bounce 的最小 tensor 数 |
| `MUSA_ENABLE_TF32` | 数值行为 | 关闭 | 允许 muDNN TF32 |

## 常用组合

### Dump GetCapability 前的图

```bash
ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID=1 \
ORT_MUSA_DUMP_GET_CAPABILITY_GRAPH_MERMAID_PATH=/tmp/musa_get_capability_graph_{}.mmd \
./.venv/bin/python your_script.py
```

### Dump runtime execution graph

```bash
ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID=1 \
ORT_MUSA_DUMP_RUNTIME_GRAPH_MERMAID_PATH=/tmp/musa_runtime_execution_graph.mmd \
./.venv/bin/python your_script.py
```

### 定位异步 kernel 错误

```bash
MUSA_EP_TRACE_KERNELS=1 \
MUSA_EP_TRACE_SYNC=1 \
./.venv/bin/python your_script.py
```

### 限制设备并启用 allocator cache

```bash
MUSA_VISIBLE_DEVICES=0 \
ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB=512 \
./.venv/bin/python your_script.py
```
