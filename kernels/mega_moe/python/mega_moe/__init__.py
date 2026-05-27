"""mega_moe —— Python entry point (calls the C++/CUDA kernel via TVM FFI).

Keeps the Python/JIT workflow, but the binding goes through tvm_ffi (no
dependency on pybind11 / torch C++ extensions). torch tensors are passed to
C++'s tvm::ffi::TensorView zero-copy via DLPack.

Generic loading/profiler utilities come from mega_common
(common/python/mega_common); this module only holds the mega_moe-specific
config and call wrappers.

Usage::

    import torch, mega_moe
    mod = mega_moe.load()                       # = mega_common.load()
    cfg = mega_moe.MoEConfig(num_ranks=1)
    y = torch.empty((num_tokens, cfg.hidden), dtype=torch.bfloat16, device='cuda')
    mod.mega_moe(y, l1_w, l1_sf, l2_w, l2_sf, sym_buffer, peer_ptrs,
                 cfg.meta(num_tokens), activation_clamp, fast_math, profiler_buf)
"""
from __future__ import annotations

from dataclasses import dataclass

# Reuse mega_common for shared components; the MEGA_KERNEL_LIB env var points to libmega_moe_ffi.so
from mega_common import load, alloc_profiler_buffer, dump_profiler_buffer  # noqa: F401

# Number of warp-roles in mega_moe (the profiler's group dimension), matching include/mega_moe/events.h
NUM_ROLES = 6


@dataclass
class MoEConfig:
    """Python mirror corresponding to C++ include/mega_moe/shapes.h::MoEConfig."""
    num_max_tokens_per_rank: int = 8192
    hidden: int = 7168
    intermediate_hidden: int = 3072
    num_experts: int = 384
    num_topk: int = 6
    num_ranks: int = 1
    rank: int = 0
    num_sms: int = 148

    def meta(self, num_tokens: int):
        """Pack into the int64 meta tensor expected by the C++ side (order: see bindings/mega_moe_ffi.cu)."""
        import torch
        return torch.tensor(
            [num_tokens, self.num_max_tokens_per_rank, self.hidden,
             self.intermediate_hidden, self.num_experts, self.num_topk,
             self.num_ranks, self.rank, self.num_sms],
            dtype=torch.int64, device="cuda")
