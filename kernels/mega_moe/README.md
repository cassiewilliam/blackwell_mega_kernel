# mega_moe — SM100 FP8×FP4 Mega-MoE

The MoE sub-project of `blackwell_mega_kernel`, refactored from DeepGEMM's
`sm100_fp8_fp4_mega_moe.cuh` (a 1644-line monolithic kernel): make the whole MoE FFN
pipeline clear, split it cleanly, build it standalone, unit-test it, and add per-SM
performance visualization.

> Shared infrastructure (profiler / Perfetto export / tvm-ffi loader) lives at the repo
> root — see the top-level [README](../../README.md) and [`common/`](../../common).

## What it is

Mega-MoE fuses the **five phases of a MoE expert layer into a single persistent kernel**,
resident on all SMs at once with warp-specialized roles:

```
        ┌──────────────────────── single kernel, persistent on all SMs ─────────────────────────┐
 input  │  ① Dispatch      ② Linear1        ③ SwiGLU            ④ Linear2       ⑤ Combine  │  output
 x(FP8) │  NVLink pull  →  x @ W1ᵀ (FP8×FP4) → silu(g)·u·w → FP8 → s @ W2ᵀ (FP8×FP4) → NVLink push → top-k reduce │  y(BF16)
 topk   │  this rank's    →  [gate ‖ up]       online amax + cast → BF16          back to src rank      │
        └────────────────────────────────────────────────────────────────────────────────┘
```

**Output-Stationary**: the tokens of one pool block are computed on the **same SM** for
both Linear1 and Linear2, eliminating cross-SM reduction — only a per-block arrival
(L1 count / L2 mask) is needed for k-wise synchronization.

**Gate + top-k are out of scope** (as in the original): the caller passes precomputed
`topk_idx` / `topk_weights`.

## Target shape (Qwen3.5, default config)

| Parameter | Value |
|---|---|
| `hidden` (H) | 7168 |
| `intermediate_hidden` (I) | 3072 |
| `num_experts` | 384 (64 per rank × 6 ranks) |
| `num_topk` | 6 |
| `num_max_tokens_per_rank` | 8192 |
| Quantization | input FP8 (E4M3, per-32 UE8M0 SF), weights FP4 (per-32 UE8M0 SF) |

See [include/mega_moe/shapes.h](include/mega_moe/shapes.h).

## Layout

```
kernels/mega_moe/
├── include/mega_moe/
│   ├── mega_moe.h          Public API: launch_mega_moe(...)
│   ├── shapes.h            Compile-time config / Qwen3.5 traits
│   ├── workspace.h         Host-side workspace / symmetric-buffer sizing
│   ├── events.h            Perfetto event ids + warp-role definitions (pairs with ../../common/include/mega/profiler.cuh)
│   └── detail/             Implementation details (to be vendored, see below)
│       ├── layout.cuh          ← DeepGEMM layout/mega_moe.cuh
│       ├── scheduler.cuh       ← DeepGEMM scheduler/mega_moe.cuh
│       ├── barrier.cuh         ← DeepGEMM comm/barrier.cuh
│       ├── tma_desc.cuh        TMA descriptor wrappers
│       ├── mma_sm100_fp8fp4.cuh FP8×FP4 UMMA thin wrappers
│       ├── tcgen05_ptx.cuh     tmem alloc/load/store
│       ├── nvlink_pull.cuh     dispatch communication primitives
│       ├── nvlink_push.cuh     combine communication primitives
│       ├── swiglu_fp4_cast.cuh L1 epilogue + online amax + FP8 cast
│       └── grouped_gemm.cuh    Linear1/Linear2 shared GEMM main loop
├── src/
│   ├── mega_moe_sm100.cu       Main kernel (five phases split into phase_* __device__ functions)
│   ├── mega_moe_launch.cu      Host: TMA descriptor setup + kernel launch
│   └── weight_transform.cu     L1 interleave + SF transpose-for-UTCCP
├── bindings/mega_moe_ffi.cu    TVM FFI C++ binding (TensorView + export macro)
├── python/mega_moe/__init__.py Kernel-specific config (reuses mega_common.load)
├── tests/                      reference_cpu.{h,cc} + test_layout.cu
└── bench/                      bench_mega_moe.cu
```

## Build (standalone)

```bash
# Host reference + unit tests only
cmake -S kernels/mega_moe -B build && cmake --build build -j && ctest --test-dir build
# Or build everything from the repo root (recommended): see ../../README.md
```

CUDA / FFI / profiler toggles reuse the top-level global options: `MEGA_BUILD_KERNEL` /
`MEGA_BUILD_FFI` / `MEGA_ENABLE_PROFILER` / `MEGA_CUTLASS_DIR` / `MEGA_TVM_FFI_DIR`.

## TVM FFI binding at a glance
[bindings/mega_moe_ffi.cu](bindings/mega_moe_ffi.cu): arguments are `tvm::ffi::TensorView`
(`.data_ptr()/.dtype()/.device()`), the stream comes from `TVMFFIEnvGetStream`, and the
export is `TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe, MegaMoE)`. From Python:
`mega_moe.load().mega_moe(y, l1_w, ...)`, with torch tensors crossing via DLPack zero-copy.

## per-SM Perfetto tracing at a glance
The generic probe is in [`../../common/include/mega/profiler.cuh`](../../common/include/mega/profiler.cuh);
mega_moe's events/roles are in [include/mega_moe/events.h](include/mega_moe/events.h)
(group = warp role: dispatch/tma_a/tma_b/mma/epilogue/combine). After a run, dump the
buffer and run `python ../../common/tools/export_perfetto.py prof.bin -o trace.json`, then
open it at https://ui.perfetto.dev.

## Mapping to the original

| This project | DeepGEMM source |
|---|---|
| `src/mega_moe_sm100.cu` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` |
| `include/mega_moe/detail/layout.cuh` | `layout/mega_moe.cuh` |
| `include/mega_moe/detail/scheduler.cuh` | `scheduler/mega_moe.cuh` |
| `include/mega_moe/detail/barrier.cuh` | `comm/barrier.cuh` |
| `src/weight_transform.cu` | `deep_gemm/mega/__init__.py`'s `_interleave_l1_weights` / `_transpose_sf_for_utccp` |
| `tests/` | `tests/test_mega_moe.py` (de-PyTorch'd) |

## Status

- [x] Project scaffold + public API + CPU reference (ctest passing)
- [x] TVM FFI binding skeleton + per-SM Perfetto probe + export tool
- [ ] Vendor `detail/` headers (layout/scheduler/barrier/tma/mma/tcgen05)
- [ ] Split the main kernel into five phases + profiler event anchors
- [ ] Single-GPU stub end-to-end (called via tvm-ffi, checked against reference_cpu)
- [ ] Multi-GPU NVLink (SymmBuffer + dispatch/combine)
```
