# common/vendor — third-party vendored code

## deep_gemm/

Low-level SM100 headers copied **verbatim** from
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (MIT License, Copyright (c) 2025
DeepSeek — see [deep_gemm/LICENSE](deep_gemm/LICENSE)). This is the complete transitive
include closure of `sm100_fp8_fp4_mega_moe.cuh` (17 files / ~3.7k LOC).

These are Blackwell low-level primitives (PTX wrappers, UMMA, TMA, SF layout,
grid/cluster barriers). Rewriting them is low-value and high-risk, so they are vendored
rather than refactored — the clean-rewrite effort goes into the kernel split, the host
launcher (de-JIT/de-torch), the tvm-ffi binding, and the profiler instead.

The `deep_gemm/` subdirectory namespace is preserved so the internal
`#include <deep_gemm/...>` directives resolve unchanged; just add `common/vendor` to the
include path at build time.

| Subdir | Contents |
|---|---|
| `comm/barrier.cuh` | grid / cluster barriers (bit31-flip + ld.acquire) |
| `common/` | math / tma_copy / types / utils / exception / compile / cute_tie |
| `layout/` | mega_moe workspace layout + sym_buffer (NVLink symmetric addressing) |
| `mma/sm100.cuh` | UMMA descriptor + policy |
| `ptx/` | inline PTX for ld_st / tcgen05 / tma / utils |
| `scheduler/mega_moe.cuh` | wave-based expert scheduling |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | the main kernel (device template, to be split/refactored) |

## csrc/

The DeepGEMM **host-side JIT + launcher slice** (24 files / ~3.3k LOC), also verbatim and
MIT. This is the transitive include closure of `csrc/apis/mega.hpp`: the NVCC-based JIT
runtime (`jit/`), the MoE heuristics + launcher + TMA-descriptor builders
(`jit_kernels/`), and shared `utils/`. The `csrc/` subtree is preserved so the internal
relative includes (`"../jit/…"`) resolve; `<deep_gemm/…>` resolves via `common/vendor` on
the include path.

We keep the JIT path (kernel is NVCC-compiled at runtime) and the torch-based launcher
**unchanged**. The only new host code is a thin TVM-FFI bridge that converts DLPack
`TensorView` → `torch::Tensor` (`torch::from_blob`) and calls
`deep_gemm::mega::{get_block_m_for_mega_moe, get_symm_buffer_size_for_mega_moe,
fp8_fp4_mega_moe}` — replacing the 28-line pybind11 `python_api.cpp`.

> To upgrade: re-`cp` the vendored files from the corresponding DeepGEMM commit; do not
> hand-edit logic here. This project's only kernel change is `#ifdef MEGA_ENABLE_PROFILER`
> guarded profiler probes (zero-cost when off, so the kernel stays byte-identical upstream).
