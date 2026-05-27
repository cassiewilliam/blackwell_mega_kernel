# mega_moe — SM100 FP8×FP4 Mega-MoE

`blackwell_mega_kernel` 下的 MoE 子工程，从 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 的
`sm100_fp8_fp4_mega_moe.cuh`（1644 行单文件大内核）重构而来：把整条 MoE FFN 管线
讲清楚、拆干净、能独立编译、单元测试、并带 per-SM 性能可视化。

> 共享基础设施（profiler / Perfetto 导出 / tvm-ffi 加载）见仓库根 [README](../../README.md) 与 [`common/`](../../common)。

## 这是什么

Mega-MoE 把一个 MoE 专家层的**五个阶段融合进同一个 persistent kernel**，
所有 SM 同时驻留、warp-specialized 分工：

```
        ┌─────────────────────────── 单 kernel，所有 SM 持久驻留 ───────────────────────────┐
 输入   │  ① Dispatch      ② Linear1        ③ SwiGLU            ④ Linear2       ⑤ Combine  │  输出
 x(FP8) │  NVLink pull  →  x @ W1ᵀ (FP8×FP4) → silu(g)·u·w → FP8 → s @ W2ᵀ (FP8×FP4) → NVLink push → top-k 规约 │  y(BF16)
 topk   │  本 rank 的     →  [gate ‖ up]       在线 amax + cast   → BF16          写回源 rank          │
        └────────────────────────────────────────────────────────────────────────────────┘
```

**Output-Stationary**：同一个 pool block 的 token 在 Linear1 / Linear2 都由
**同一个 SM** 算完，消除跨 SM 规约——只需 per-block arrival 做 k-wise 同步。

**Gate + top-k 不在本工程内**（与原版一致）：调用者传入预先算好的 `topk_idx` / `topk_weights`。

## 目标形状（Qwen3.5，默认 config）

| 参数 | 值 |
|---|---|
| `hidden` (H) | 7168 |
| `intermediate_hidden` (I) | 3072 |
| `num_experts` | 384（每 rank 64 × 6 ranks）|
| `num_topk` | 6 |
| `num_max_tokens_per_rank` | 8192 |
| 量化 | 输入 FP8 (E4M3, per-32 UE8M0 SF)，权重 FP4 (per-32 UE8M0 SF) |

见 [include/mega_moe/shapes.h](include/mega_moe/shapes.h)。

## 工程布局

```
kernels/mega_moe/
├── include/mega_moe/
│   ├── mega_moe.h          公开 API：launch_mega_moe(...)
│   ├── shapes.h            编译期 config / Qwen3.5 traits
│   ├── workspace.h         host-side workspace / symm-buffer 尺寸计算
│   ├── events.h            Perfetto 事件 id + warp-role 定义（配 ../../common/include/mega/profiler.cuh）
│   └── detail/             实现细节（待 vendor，见下）
│       ├── layout.cuh          ← DeepGEMM layout/mega_moe.cuh
│       ├── scheduler.cuh       ← DeepGEMM scheduler/mega_moe.cuh
│       ├── barrier.cuh         ← DeepGEMM comm/barrier.cuh
│       ├── tma_desc.cuh        TMA descriptor 封装
│       ├── mma_sm100_fp8fp4.cuh FP8×FP4 UMMA 薄封装
│       ├── tcgen05_ptx.cuh     tmem alloc/load/store
│       ├── nvlink_pull.cuh     dispatch 通信原语
│       ├── nvlink_push.cuh     combine 通信原语
│       ├── swiglu_fp4_cast.cuh L1 epilogue + 在线 amax + FP8 cast
│       └── grouped_gemm.cuh    Linear1/Linear2 共享 GEMM 主循环
├── src/
│   ├── mega_moe_sm100.cu       主 kernel（五段拆成 phase_* __device__ 函数）
│   ├── mega_moe_launch.cu      host：TMA descriptor + kernel launch
│   └── weight_transform.cu     L1 interleave + SF transpose-for-UTCCP
├── bindings/mega_moe_ffi.cu    TVM FFI C++ 绑定（TensorView + 导出宏）
├── python/mega_moe/__init__.py kernel 专属 config（复用 mega_common.load）
├── tests/                      reference_cpu.{h,cc} + test_layout.cu
└── bench/                      bench_mega_moe.cu
```

## 构建（standalone）

```bash
# 仅 host 参考 + 单元测试
cmake -S kernels/mega_moe -B build && cmake --build build -j && ctest --test-dir build
# 或从仓库根统一构建（推荐）：见 ../../README.md
```

CUDA / FFI / profiler 的开关沿用顶层全局选项：`MEGA_BUILD_KERNEL` / `MEGA_BUILD_FFI` /
`MEGA_ENABLE_PROFILER` / `MEGA_CUTLASS_DIR` / `MEGA_TVM_FFI_DIR`。

## TVM FFI 绑定速览
[bindings/mega_moe_ffi.cu](bindings/mega_moe_ffi.cu)：参数 `tvm::ffi::TensorView`（`.data_ptr()/.dtype()/.device()`），
stream 经 `TVMFFIEnvGetStream`，导出 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe, MegaMoE)`。
Python：`mega_moe.load().mega_moe(y, l1_w, ...)`，torch tensor 走 DLPack 零拷贝。

## per-SM Perfetto tracing 速览
通用探针在 [`../../common/include/mega/profiler.cuh`](../../common/include/mega/profiler.cuh)，
mega_moe 的事件/角色在 [include/mega_moe/events.h](include/mega_moe/events.h)（group = warp-role：
dispatch/tma_a/tma_b/mma/epilogue/combine）。跑完 dump buffer →
`python ../../common/tools/export_perfetto.py prof.bin -o trace.json` → 拖进 https://ui.perfetto.dev。

## 与原版的对应关系

| 本工程 | DeepGEMM 来源 |
|---|---|
| `src/mega_moe_sm100.cu` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` |
| `include/mega_moe/detail/layout.cuh` | `layout/mega_moe.cuh` |
| `include/mega_moe/detail/scheduler.cuh` | `scheduler/mega_moe.cuh` |
| `include/mega_moe/detail/barrier.cuh` | `comm/barrier.cuh` |
| `src/weight_transform.cu` | `deep_gemm/mega/__init__.py` 的 `_interleave_l1_weights` / `_transpose_sf_for_utccp` |
| `tests/` | `tests/test_mega_moe.py`（去 PyTorch 化）|

## 状态

- [x] 工程骨架 + 公开 API + CPU 参考（ctest 通过）
- [x] TVM FFI 绑定层骨架 + per-SM Perfetto 探针 + 导出工具
- [ ] vendor `detail/` 头文件（layout/scheduler/barrier/tma/mma/tcgen05）
- [ ] 主 kernel 五段拆分 + profiler 事件 anchor
- [ ] 单 GPU stub 端到端（经 tvm-ffi 调通，与 reference_cpu 对拍）
- [ ] multi-GPU NVLink（SymmBuffer + dispatch/combine）
