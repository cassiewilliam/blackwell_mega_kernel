# blackwell_mega_kernel

SM100 (Blackwell) **mega kernel 集合** —— 把多个"单 kernel 融合整条管线"的大算子，
重构成干净、可独立编译、单元可测、并自带 **per-SM Perfetto 时间线可视化** 的工程。

每个 mega kernel 在 [`kernels/`](kernels/) 下各占一个子目录，共享 [`common/`](common/)
里的基础设施（profiler 探针、Perfetto 导出、TVM FFI 加载）。

## 当前内容

| Kernel | 说明 | 状态 |
|---|---|---|
| [`kernels/mega_moe`](kernels/mega_moe) | FP8×FP4 MoE 五段融合（Dispatch→L1→SwiGLU→L2→Combine），重构自 DeepGEMM `sm100_fp8_fp4_mega_moe.cuh` | 骨架 + CPU 参考 + FFI/profiler 绑定 ✅；kernel 实现进行中 |

> 后续可能加入其它 mega kernel（如 mega_ffn / mega_attention），结构已为此预留。

## 仓库布局

```
blackwell_mega_kernel/
├── CMakeLists.txt                  顶层：全局选项 + add_subdirectory(kernels/*)
├── common/                         跨 kernel 共享
│   ├── include/mega/
│   │   └── profiler.cuh            per-SM Perfetto 探针（device 宏，零开销可关）
│   ├── python/mega_common/
│   │   └── __init__.py             tvm_ffi 模块加载 + profiler buffer 分配
│   └── tools/
│       └── export_perfetto.py      profiler buffer → Perfetto trace JSON
└── kernels/
    └── mega_moe/                   见 kernels/mega_moe/README.md
        ├── CMakeLists.txt          可被顶层引入，也可 standalone 构建
        ├── include/mega_moe/       公开 API + shapes + workspace + events
        ├── bindings/               TVM FFI C++ 绑定
        ├── python/mega_moe/        kernel 专属 config + 调用封装
        ├── tests/                  五段 CPU 黄金参考 + 单元测试
        ├── src/                    CUDA kernel（进行中）
        └── bench/                  性能基线（进行中）
```

## 三个设计支柱

1. **干净的 C++/CUDA 内核**：单文件大内核拆成可读的 `phase_*` 函数 + warp-role 表。
2. **TVM FFI 绑定**：经 `tvm::ffi` C++ 接口暴露稳定 ABI，Python（torch→DLPack 零拷贝）
   可调用——保留 Python/JIT 工作流，不依赖 pybind11 / torch C++ 扩展。
3. **per-SM Perfetto tracing**（仿 [FlashInfer profiler](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/profiler.cuh)）：
   `-DMEGA_ENABLE_PROFILER` 开关，关闭零开销；按 SM 导出时间线看各段重叠。

## 构建

```bash
# 仅 host 参考 + 单元测试（无需 GPU/CUTLASS/tvm-ffi）
cmake -B build && cmake --build build -j && ctest --test-dir build

# 含 CUDA kernel + tvm-ffi 绑定 + profiler（需 B200 + CUTLASS + apache-tvm-ffi）
cmake -B build \
  -DMEGA_BUILD_KERNEL=ON -DMEGA_BUILD_FFI=ON -DMEGA_ENABLE_PROFILER=ON \
  -DMEGA_CUTLASS_DIR=/path/to/cutlass/include \
  -DMEGA_TVM_FFI_DIR=$(python -c "import tvm_ffi,os;print(os.path.dirname(tvm_ffi.__file__))")
cmake --build build -j
```
