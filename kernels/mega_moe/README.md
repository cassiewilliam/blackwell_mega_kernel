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

Config/heuristics (block sizes, stages) live in
[src/csrc/jit_kernels/heuristics/mega_moe.hpp](src/csrc/jit_kernels/heuristics/mega_moe.hpp).

## Layout

The MegaMoE-specific source (derived from DeepGEMM, **editable — modify here**) lives in
`src/`; shared DeepGEMM infrastructure stays vendored in `../../common/vendor/`.

```
kernels/mega_moe/
├── src/                              EDITABLE MegaMoE source (kept consistent w/ DeepGEMM)
│   ├── deep_gemm/                      device code
│   │   ├── impls/sm100_fp8_fp4_mega_moe.cuh   the kernel (+ #ifdef-guarded profiler probes)
│   │   ├── layout/mega_moe.cuh                workspace / sym-buffer layout
│   │   └── scheduler/mega_moe.cuh             wave scheduler
│   └── csrc/                           host code
│       ├── apis/mega.hpp                       top-level API (fp8_fp4_mega_moe)
│       ├── jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp   NVCC-JIT launcher
│       └── jit_kernels/heuristics/mega_moe.hpp  config heuristics (block_m, stages, …) — tune here
├── bindings/mega_moe_ffi.cu          TVM FFI bridge (DLPack TensorView → torch → launcher)
├── python/mega_moe/__init__.py       kernel-specific config (reuses mega_common.load)
├── tests/test_e2e.py                 multi-rank end-to-end test (vs deep_gemm)
└── build_ffi.sh                      builds the .so + merged include trees (jit_root/host_root)
```

> Shared (don't modify): `common/vendor/deep_gemm/{comm,common,mma,ptx,layout/sym_buffer}` +
> `common/vendor/csrc/{jit,utils,jit_kernels/...}`. `build_ffi.sh` builds **merged include
> trees** so `<deep_gemm/...>` and `csrc/...` resolve the MegaMoE files to `src/` and
> everything else to `common/vendor/`.

## Build / run (B200 container)

```bash
bash kernels/mega_moe/build_ffi.sh                 # -> build_ffi/{libmega_moe_ffi.so, jit_root}
MEGA_MOE_LIB=build_ffi/libmega_moe_ffi.so MEGA_JIT_ROOT=build_ffi/jit_root \
  python kernels/mega_moe/tests/test_e2e.py        # multi-rank; checks vs deep_gemm

# edit the kernel → rebuild → JIT picks up your src/ version
$EDITOR src/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh && bash kernels/mega_moe/build_ffi.sh
```

Profiling: see [../../docs/profiling.md](../../docs/profiling.md).

## TVM FFI binding at a glance
[bindings/mega_moe_ffi.cu](bindings/mega_moe_ffi.cu): arguments are `tvm::ffi::TensorView`
(`.data_ptr()/.dtype()/.device()`), the stream comes from `TVMFFIEnvGetStream`, and the
export is `TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe, MegaMoE)`. From Python:
`mega_moe.load().mega_moe(y, l1_w, ...)`, with torch tensors crossing via DLPack zero-copy.

## per-SM Perfetto tracing at a glance
The generic probe is in [`../../common/include/mega/profiler.cuh`](../../common/include/mega/profiler.cuh);
the kernel's `#ifdef MEGA_ENABLE_PROFILER` probes are in
[src/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh](src/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)
(warp roles: dispatch / tma-a / tma-b / mma / epilogue; per-tile L1/L2/Act/Combine with
`-DMEGA_PROFILE_BLOCKS`). Dump the buffer, then
`python ../../common/tools/export_perfetto.py prof.bin -o trace.json` → open at
https://ui.perfetto.dev. **Full guide: [../../docs/profiling.md](../../docs/profiling.md).**

## Provenance

Derived from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 2.5.0 (MIT). The MegaMoE
kernel + JIT launcher + heuristics are kept verbatim (editable in `src/`); shared infra is
vendored under `../../common/vendor/`. Only additions: the TVM-FFI bridge and the
`#ifdef`-guarded profiler probes.

## Status

- [x] Vendored DeepGEMM (latest 2.5.0+714dd1a): kernel + JIT + launcher, unchanged
- [x] **TVM-FFI bridge** — `mega_moe()` callable from Python; JIT-compiles the kernel
- [x] **End-to-end on B200** — output bit-identical to `deep_gemm.fp8_fp4_mega_moe`
      (`max|ours-ref| = 0`), via `tests/test_e2e.py`
- [x] **per-SM Perfetto profiler** — 148 SMs × begin/end spans → trace JSON
      (`MEGA_PROF=1 DG_JIT_EXTRA_FLAGS=-DMEGA_ENABLE_PROFILER`), zero-cost when off
- [x] **Multi-GPU NVLink** — real multi-rank EP (4/8 ranks) PASS vs deep_gemm (`diff=0`)
- [x] **per-tile profiler** — per-role + L1/L2/Act/Combine lanes (`-DMEGA_PROFILE_BLOCKS`),
      see [../../docs/profiling.md](../../docs/profiling.md)
- [ ] Integrate as SonicMoE's forward (EP-training fusion — the project goal)
