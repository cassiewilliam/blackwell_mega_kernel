"""mega_moe —— Python 入口（经 TVM FFI 调用 C++/CUDA kernel）。

保留 Python/JIT 工作流，但绑定走 tvm_ffi（不依赖 pybind11 / torch C++ 扩展）。
torch tensor 通过 DLPack 零拷贝传给 C++ 的 tvm::ffi::TensorView。

通用加载/profiler 工具来自 mega_common（common/python/mega_common），本模块只放
mega_moe 专属的 config 与调用封装。

用法::

    import torch, mega_moe
    mod = mega_moe.load()                       # = mega_common.load()
    cfg = mega_moe.MoEConfig(num_ranks=1)
    y = torch.empty((num_tokens, cfg.hidden), dtype=torch.bfloat16, device='cuda')
    mod.mega_moe(y, l1_w, l1_sf, l2_w, l2_sf, sym_buffer, peer_ptrs,
                 cfg.meta(num_tokens), activation_clamp, fast_math, profiler_buf)
"""
from __future__ import annotations

from dataclasses import dataclass

# 通用件复用 mega_common；MEGA_KERNEL_LIB 环境变量指向 libmega_moe_ffi.so
from mega_common import load, alloc_profiler_buffer, dump_profiler_buffer  # noqa: F401

# mega_moe 的 warp-role 数（profiler 的 group 维度），与 include/mega_moe/events.h 一致
NUM_ROLES = 6


@dataclass
class MoEConfig:
    """与 C++ include/mega_moe/shapes.h::MoEConfig 对应的 Python 镜像。"""
    num_max_tokens_per_rank: int = 8192
    hidden: int = 7168
    intermediate_hidden: int = 3072
    num_experts: int = 384
    num_topk: int = 6
    num_ranks: int = 1
    rank: int = 0
    num_sms: int = 148

    def meta(self, num_tokens: int):
        """打包成 C++ 侧约定的 int64 meta tensor（顺序见 bindings/mega_moe_ffi.cu）。"""
        import torch
        return torch.tensor(
            [num_tokens, self.num_max_tokens_per_rank, self.hidden,
             self.intermediate_hidden, self.num_experts, self.num_topk,
             self.num_ranks, self.rank, self.num_sms],
            dtype=torch.int64, device="cuda")
