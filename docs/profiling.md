# Per-SM Perfetto Profiling for MegaMoE

In-kernel software profiling (FlashInfer-style) that records, per SM and per warp-role,
when each phase of the fused MegaMoE kernel runs — exported to a [Perfetto](https://ui.perfetto.dev)
timeline. Two views are provided:

1. **per-role (default)** — one faithful span per warp-role per SM (Dispatch / TMA-A /
   TMA-B / MMA / Epilogue). Use this for **absolute timing and overlap**.
2. **PR-style 3-lane (opt-in)** — per-tile slices laid out like PR #316's "(c) Ours" figure:
   `Dispatch | Computation (L1·L2) | Act&Combine (Act·Combine)`. Use this for **structure /
   per-wave rhythm** (it perturbs absolute timing — see caveats).

Everything is gated by compile flags; with profiling off the kernel is byte-identical to
upstream DeepGEMM.

## Build flags

Injected into the runtime NVCC JIT via the `DG_JIT_EXTRA_FLAGS` env hook (added to the
vendored `compiler.hpp`):

| Flag | Effect |
|---|---|
| `-DMEGA_ENABLE_PROFILER` | enable probes; **coarse per-role** spans. Zero-cost when absent. |
| `-DMEGA_PROFILE_BLOCKS` | additionally emit **per-tile** L1/L2 (in MMA loop) and Act/Combine (in epilogue loop). Coarse MMA/Epilogue spans are gated off in this mode. |
| `-DMEGA_PROFILER_USE_GLOBALTIMER` | timestamp with `%globaltimer` (cross-SM synchronized) instead of the default `clock64`. |

## Timestamp source: `clock64` vs `%globaltimer`

The probe timestamp defaults to **`clock64()`** (SM clock, a few cycles to read, per-SM —
not cross-SM synchronized). `%globaltimer` is cross-SM synchronized but **high-latency and
serializes the warp pipeline**: with per-block probes in the MMA hot loop it inflated the
kernel ~**17×** (568µs → ~10ms). With `clock64` the overhead drops to ~**1.27×**. So:

- per-role (few probes): either source is fine; `%globaltimer` gives true cross-SM alignment.
- per-block (hot-loop probes): use `clock64` (default). Timestamps are in **SM cycles**
  (~1.8 GHz on B200, so `cycles / 1800 ≈ µs`); good for per-SM-track relative timing.

## Running (B200 container)

```bash
# 1) build the bridge .so + the merged JIT include root
bash kernels/mega_moe/build_ffi.sh

# 2a) per-role profiling (faithful timing)
export MEGA_MOE_LIB=build_ffi/libmega_moe_ffi.so MEGA_JIT_ROOT=build_ffi/jit_root
export MEGA_PROF=1 DG_JIT_EXTRA_FLAGS=-DMEGA_ENABLE_PROFILER
python kernels/mega_moe/tests/test_e2e.py            # rank 0 dumps prof.bin

# 2b) PR-style per-tile lanes (structure)
export DG_JIT_EXTRA_FLAGS="-DMEGA_ENABLE_PROFILER -DMEGA_PROFILE_BLOCKS"
python kernels/mega_moe/tests/test_e2e.py

# 3) decode (text summary) and export (Perfetto JSON)
python common/tools/decode_prof.py prof.bin
python common/tools/export_perfetto.py prof.bin -o trace.json \
    --events Dispatch,TMA-A,TMA-B,MMA,Epilogue,L1,L2,Act,Combine \
    --roles 0,5,6,7,8 --max-sms 6     # 3-lane PR view, a few SMs
# open trace.json at https://ui.perfetto.dev
```

`MegaMoE is collective`: the profiling kernel call must run on **all ranks** together
(dispatch/combine pull from / push to peer ranks); only rank 0 captures the buffer. The
test handles this.

### exporter options (`common/tools/export_perfetto.py`)
- `--events a,b,c` — event-id → name table (order = event_idx).
- `--roles 5,6` — keep only these event_idx (e.g. L1/L2 only; drops coarse spans so the
  view auto-scales to the compute tiles).
- `--max-sms N` — keep the first N SMs (smaller file; SMs run in lockstep, so a few suffice).
  The smc channel corrupts large single-shot transfers, so prefer small files.

## How the buffer is laid out

8-byte entries. `entry[0]` = header `{num_blocks, num_groups}`. Each event:
`tag` (low 32: `[1:0]`=type, `[11:2]`=event_idx, `[23:12]`=block, `[31:24]`=sm) +
`delta_time` (high 32). Slot for a `(block, group, cursor)` event:
`1 + (block*num_groups + group) + cursor*(num_blocks*num_groups)`; begin = even cursor,
end = odd. Probes are **stateless** (re-derive blockIdx/gridDim/smid each call, take the
buffer pointer as an argument) — required because the kernel runs warp-specialized
`reg_reconfig`/`setmaxnreg`, which corrupts any register state cached across the kernel body.

## Findings (4 GPU, EP4, Flash-per-rank: 32 experts/rank, H4096, I2048, top-6, bs512)

**per-role (faithful, ~clock cycles):** Dispatch ≈ Epilogue ≈ whole kernel (~457k cyc),
TMA-A/B ≈ 427k, MMA leader ≈ 178k (non-leader ≈ 160 cyc). All roles overlap nearly the
full kernel — warp-specialization keeps communication, feed, compute, and epilogue running
concurrently.

**per-tile (PR-style):** per leader-SM, ~7 L1 + ~7 L2 tiles, ~7 Act + ~7 Combine:

| tile | per-tile (cyc) | note |
|---|---|---|
| L1 (x@W1 → gate‖up) | ~11.5k | 2× L2 (outputs 2N) |
| L2 (swiglu@W2) | ~5.9k | |
| **Act** (SwiGLU + amax + FP8 cast) | ~12.0k | **biggest single phase** — ≈ L1 GEMM |
| Combine (NVLink push + topk reduce) | ~5.3k | ≈ L2 |

**Key insight:** in an area selection, **Act has the largest summed time** (e.g. 223µs vs
L1 107µs vs Combine 85µs vs L2 47µs). The L1-epilogue (SiLU·up·weight + online amax + FP8
cast) is the compute hotspot at this size, not the GEMMs — optimize there first.

**2-CTA structure:** only ~half the SMs (cluster leaders) execute compute tiles; partner
CTAs' MMA warp is idle (~160 cyc). Visible as empty Computation/Act&Combine lanes on
odd SMs.

## PR #316 benchmark reproduction (EP4 ≈ EP8 when compute-bound)

`tests/test_mega_moe.py` from DeepGEMM is the benchmark. Reproduced on 4 idle B200s (EP4,
per-rank workload matched to PR's EP8). At large batch (GEMM-bound) EP4 matches PR's EP8
within ~2–5%:

| config | batch | EP4 (ours) | PR EP8 |
|---|---|---|---|
| V4-Pro (H7168/I3072, 48 exp/rank) | 8192 | 2884µs / 2263 TF | 2818µs / 2304 TF |
| V4-Flash (H4096/I2048, 32 exp/rank) | 8192 | 1344µs / 1840 TF | 1283µs / 1928 TF |

Small batches are latency/overhead-bound and differ more (EP4 fewer ranks + shared-box
contention). A clean EP8 reproduction needs 8 free GPUs.

## Caveats

- **per-block probes distort absolute timing** (~1.3× even with clock64). Use them for
  structure / per-tile ratios, not for headline latency — that's the **per-role** view.
- **clock64 is per-SM** (not cross-SM synced); per-SM tracks are correct, cross-SM
  alignment is approximate (all SMs start within ~200ns anyway). Use
  `-DMEGA_PROFILER_USE_GLOBALTIMER` if exact cross-SM alignment is needed.
- The smc-toc download channel corrupts large files; keep traces small (`--max-sms`).
