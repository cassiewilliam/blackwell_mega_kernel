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

> To upgrade: re-`cp` these 17 files from the corresponding DeepGEMM commit; do not hand-edit
> logic in this directory. This project's changes (phase split / profiler anchors) live in
> `kernels/mega_moe/src/` so they never pollute the vendored tree.
