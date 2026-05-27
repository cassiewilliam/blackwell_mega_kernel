"""End-to-end test: drive our TVM-FFI MegaMoE bridge on a B200, single rank.

Uses deep_gemm ONLY for input prep (FP8/FP4 quant, weight transform, SymmBuffer);
the kernel call goes through OUR libmega_moe_ffi.so. Cross-checks our output against
deep_gemm's own fp8_fp4_mega_moe (same vendored kernel → should match).

Run inside the container:
    MEGA_MOE_LIB=build_ffi/libmega_moe_ffi.so MEGA_JIT_ROOT=build_ffi/jit_root \
        python kernels/mega_moe/tests/test_e2e.py
"""
import os
import sys
import torch
import torch.multiprocessing as mp

# common/python on path for mega_common (profiler buffer + perfetto helpers)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "python"))
import mega_common

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist

import tvm_ffi

H, I = 512, 512          # hidden, intermediate; dim/32 must be 16B-aligned (=> dim % 512 == 0)
E, TK = 8, 2             # experts, top-k
NUM_MAX = 128            # max tokens/rank (aligned up to block_m internally)
NUM_TOKENS = 64


def cast_grouped_weights_to_fp4(bf16_w):
    g, n, k = bf16_w.shape
    w = torch.empty((g, n, k // 2), device="cuda", dtype=torch.int8)
    sf = torch.empty((g, n, k // 32), device="cuda", dtype=torch.float)
    for i in range(g):
        w[i], sf[i] = per_token_cast_to_fp4(bf16_w[i], use_ue8m0=True, gran_k=32)
    sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), g)
    return w, sf


def fill(buffer, x, topk_idx, topk_weights, n):
    buffer.x[:n].copy_(x[0])
    buffer.x_sf[:n].copy_(x[1])
    buffer.topk_idx[:n].copy_(topk_idx)
    buffer.topk_weights[:n].copy_(topk_weights)


def run(local_rank, num_local_ranks):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(0)
    Er = E // num_ranks

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, E, NUM_MAX, TK, H, I)

    # inputs
    x_bf = torch.randn((NUM_TOKENS, H), dtype=torch.bfloat16, device="cuda")
    l1w = torch.randn((Er, I * 2, H), dtype=torch.bfloat16, device="cuda")
    l2w = torch.randn((Er, H, I), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((NUM_TOKENS, E), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, TK, dim=-1, largest=True, sorted=False)

    x = per_token_cast_to_fp8(x_bf, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    l1 = cast_grouped_weights_to_fp4(l1w)
    l2 = cast_grouped_weights_to_fp4(l2w)
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)

    bp = buffer.handle.buffer_ptrs
    print(f"[dbg] buffer_ptrs type={type(bp)} val={bp}")

    # --- ours FIRST, on a clean filled buffer (isolate from reference) ---
    mod = tvm_ffi.load_module(os.path.abspath(os.environ["MEGA_MOE_LIB"]))
    # JIT cd's into a tmp dir before nvcc, so library_root must be ABSOLUTE.
    mod.init(os.path.abspath(os.environ["MEGA_JIT_ROOT"]),
             os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    fill(buffer, x, topk_idx, topk_weights, NUM_TOKENS)
    torch.cuda.synchronize()
    ptrs = torch.as_tensor(list(bp), dtype=torch.int64, device="cuda")
    y_ours = torch.empty((NUM_TOKENS, H), dtype=torch.bfloat16, device="cuda")
    empty_prof = torch.empty(0, dtype=torch.int64, device="cuda")  # profiler disabled
    mod.mega_moe(y_ours, tl1[0], tl1[1], tl2[0], tl2[1],
                 buffer.buffer, ptrs, rank, buffer.num_max_tokens_per_rank,
                 E, TK, float("inf"), True, empty_prof)
    torch.cuda.synchronize()
    print(f"[dbg] ours-alone nonzero={(y_ours != 0).any().item()} "
          f"absmax={y_ours.float().abs().max().item():.4g}")

    # --- optional: per-SM Perfetto profiling run (needs JIT built with -DMEGA_ENABLE_PROFILER) ---
    if os.environ.get("MEGA_PROF"):
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        prof = mega_common.alloc_profiler_buffer(num_sms, num_groups=1, max_events=8)
        fill(buffer, x, topk_idx, topk_weights, NUM_TOKENS)
        torch.cuda.synchronize()
        mod.mega_moe(y_ours, tl1[0], tl1[1], tl2[0], tl2[1],
                     buffer.buffer, ptrs, rank, buffer.num_max_tokens_per_rank,
                     E, TK, float("inf"), True, prof)
        torch.cuda.synchronize()
        nz = int((prof != 0).sum().item())
        mega_common.dump_profiler_buffer(prof, "prof.bin")
        print(f"[prof] nonzero entries={nz} -> prof.bin "
              f"(python ../../common/tools/export_perfetto.py prof.bin)")

    # --- reference: deep_gemm's own kernel ---
    fill(buffer, x, topk_idx, topk_weights, NUM_TOKENS)
    y_ref = torch.empty((NUM_TOKENS, H), dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_fp4_mega_moe(y_ref, tl1, tl2, buffer, activation_clamp=None, fast_math=True)
    torch.cuda.synchronize()

    # --- compare ---
    diff = (y_ours.float() - y_ref.float()).abs().max().item()
    print(f"[rank {rank}] y_ours shape={tuple(y_ours.shape)} "
          f"finite={torch.isfinite(y_ours).all().item()} "
          f"nonzero={(y_ours != 0).any().item()} max|ours-ref|={diff:.4g}")
    assert torch.isfinite(y_ours).all(), "non-finite output"
    assert diff < 1e-2, f"output mismatch vs deep_gemm: {diff}"
    print(f"[rank {rank}] PASS")


if __name__ == "__main__":
    mp.spawn(run, args=(1,), nprocs=1)
