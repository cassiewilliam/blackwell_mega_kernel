# blackwell_mega_kernel

A collection of SM100 (Blackwell) **mega kernels** — large operators that fuse an
entire pipeline into a single kernel — refactored into a clean project that builds
standalone, is unit-testable, and ships with **per-SM Perfetto timeline visualization**.

Each mega kernel lives in its own directory under [`kernels/`](kernels/) and shares the
infrastructure in [`common/`](common/) (profiler probes, Perfetto export, TVM FFI loader).

> **Attribution.** The kernels and host launcher in this repo are derived from
> [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (MIT License, Copyright (c) 2025
> DeepSeek). DeepGEMM code is vendored verbatim under [`common/vendor/`](common/vendor)
> (see its LICENSE/README). This repository exists for **study and further development**;
> our own additions are the TVM-FFI binding, the per-SM Perfetto profiler, and the
> integration work described below.

## Project goal

This repo is part of a **training-time Expert-Parallel (EP) optimization** built by fusing
**SonicMoE + MegaMoE**. The MoE kernels here (MegaMoE) are intended to become the high-
performance forward path of that fused MoE layer. Roadmap:

1. **Implement SonicMoE** (the EP training MoE path) first.
2. **Replace SonicMoE's forward with MegaMoE** — wire the fused FP8×FP4 MegaMoE kernel
   (exposed via the TVM-FFI binding below) in as SonicMoE's forward, with the per-SM
   Perfetto profiler for performance analysis during EP training.

The current work in this repo is step toward (2): make MegaMoE callable (TVM-FFI) and
observable (per-SM Perfetto) while keeping the DeepGEMM kernel itself unchanged.

## Contents

| Kernel | Description | Status |
|---|---|---|
| [`kernels/mega_moe`](kernels/mega_moe) | FP8×FP4 5-phase fused MoE (Dispatch→L1→SwiGLU→L2→Combine), refactored from DeepGEMM `sm100_fp8_fp4_mega_moe.cuh` | Scaffold + CPU reference + FFI/profiler bindings ✅; kernel port in progress |

> More mega kernels (e.g. mega_ffn / mega_attention) may follow; the structure is built to accommodate them.

## Repository layout

```
blackwell_mega_kernel/
├── docs/profiling.md               per-SM/per-tile profiling guide
├── common/                         shared across kernels
│   ├── include/mega/profiler.cuh   per-SM Perfetto probe (device macros, zero-cost when off)
│   ├── python/mega_common/         tvm_ffi module loading + profiler buffer helpers
│   ├── tools/                      export_perfetto.py / decode_prof.py (trace tooling)
│   └── vendor/{deep_gemm,csrc}/    vendored DeepGEMM shared infra (MIT, verbatim)
└── kernels/
    └── mega_moe/                   see kernels/mega_moe/README.md
        ├── src/                    EDITABLE MegaMoE source (deep_gemm/ device + csrc/ host)
        ├── bindings/               TVM FFI C++ binding
        ├── python/mega_moe/        kernel-specific config + call wrappers
        ├── tests/test_e2e.py       multi-rank end-to-end test (vs deep_gemm)
        └── build_ffi.sh            builds the .so + merged include trees
```

## Three design pillars

1. **Kernel kept consistent with DeepGEMM**: the MegaMoE kernel + JIT launcher are
   vendored/derived verbatim (no rewrite); only the bridge and `#ifdef`-guarded profiler
   probes are added. The editable source lives in `kernels/mega_moe/src/`.
2. **TVM FFI binding**: exposed through the `tvm::ffi` C++ interface as a stable ABI,
   callable from Python (torch→DLPack, zero-copy). Keeps the Python/JIT (NVCC) workflow.
3. **per-SM Perfetto tracing** (modeled on the
   [FlashInfer profiler](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/profiler.cuh)):
   gated by `-DMEGA_ENABLE_PROFILER`, zero-cost when off; per-role timeline or per-tile
   L1/L2/Act/Combine lanes. Full guide: [docs/profiling.md](docs/profiling.md).

## Build & run (B200 + CUTLASS + apache-tvm-ffi)

The build is driven by `build_ffi.sh` (the runtime NVCC JIT compiles the kernel; the bridge
`.so` is host-only g++). There is no CMake build.

```bash
bash kernels/mega_moe/build_ffi.sh                          # -> build_ffi/libmega_moe_ffi.so + jit_root/
MEGA_MOE_LIB=build_ffi/libmega_moe_ffi.so MEGA_JIT_ROOT=build_ffi/jit_root \
  python kernels/mega_moe/tests/test_e2e.py                 # multi-rank, checks vs deep_gemm
```
