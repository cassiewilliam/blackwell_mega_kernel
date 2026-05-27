# blackwell_mega_kernel

A collection of SM100 (Blackwell) **mega kernels** — large operators that fuse an
entire pipeline into a single kernel — refactored into a clean project that builds
standalone, is unit-testable, and ships with **per-SM Perfetto timeline visualization**.

Each mega kernel lives in its own directory under [`kernels/`](kernels/) and shares the
infrastructure in [`common/`](common/) (profiler probes, Perfetto export, TVM FFI loader).

## Contents

| Kernel | Description | Status |
|---|---|---|
| [`kernels/mega_moe`](kernels/mega_moe) | FP8×FP4 5-phase fused MoE (Dispatch→L1→SwiGLU→L2→Combine), refactored from DeepGEMM `sm100_fp8_fp4_mega_moe.cuh` | Scaffold + CPU reference + FFI/profiler bindings ✅; kernel port in progress |

> More mega kernels (e.g. mega_ffn / mega_attention) may follow; the structure is built to accommodate them.

## Repository layout

```
blackwell_mega_kernel/
├── CMakeLists.txt                  Top-level: global options + add_subdirectory(kernels/*)
├── common/                         Shared across kernels
│   ├── include/mega/
│   │   └── profiler.cuh            per-SM Perfetto probe (device macros, zero-cost when off)
│   ├── python/mega_common/
│   │   └── __init__.py             tvm_ffi module loading + profiler buffer helpers
│   ├── tools/
│   │   └── export_perfetto.py      profiler buffer → Perfetto trace JSON
│   └── vendor/deep_gemm/           Vendored DeepGEMM device headers (MIT, verbatim)
└── kernels/
    └── mega_moe/                   See kernels/mega_moe/README.md
        ├── CMakeLists.txt          Usable from the top level or standalone
        ├── include/mega_moe/       Public API + shapes + workspace + events
        ├── bindings/               TVM FFI C++ binding
        ├── python/mega_moe/        Kernel-specific config + call wrappers
        ├── tests/                  5-phase CPU golden reference + unit tests
        ├── src/                    CUDA kernel (in progress)
        └── bench/                  Performance baselines (in progress)
```

## Three design pillars

1. **Clean C++/CUDA kernel**: the single monolithic kernel is split into readable
   `phase_*` functions plus a warp-role table.
2. **TVM FFI binding**: exposed through the `tvm::ffi` C++ interface as a stable ABI,
   callable from Python (torch→DLPack, zero-copy). Keeps the Python/JIT workflow without
   depending on pybind11 or the torch C++ extension.
3. **per-SM Perfetto tracing** (modeled on the
   [FlashInfer profiler](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/profiler.cuh)):
   gated by `-DMEGA_ENABLE_PROFILER`, zero-cost when off; exports a per-SM timeline so you
   can see how the phases overlap.

## Build

```bash
# Host reference + unit tests only (no GPU/CUTLASS/tvm-ffi needed)
cmake -B build && cmake --build build -j && ctest --test-dir build

# With the CUDA kernel + tvm-ffi binding + profiler (needs B200 + CUTLASS + apache-tvm-ffi)
cmake -B build \
  -DMEGA_BUILD_KERNEL=ON -DMEGA_BUILD_FFI=ON -DMEGA_ENABLE_PROFILER=ON \
  -DMEGA_CUTLASS_DIR=/path/to/cutlass/include \
  -DMEGA_TVM_FFI_DIR=$(python -c "import tvm_ffi,os;print(os.path.dirname(tvm_ffi.__file__))")
cmake --build build -j
```
