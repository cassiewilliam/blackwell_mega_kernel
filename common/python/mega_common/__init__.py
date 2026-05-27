"""mega_common —— blackwell_mega_kernel 各 kernel 共享的 Python 基础设施。

只放与具体 kernel 无关的通用件：
  * load(): 经 tvm_ffi 加载编译好的 .so 模块
  * alloc_profiler_buffer(): 分配 per-SM Perfetto 事件 buffer
  * dump_profiler_buffer(): 把 buffer 落盘，交给 common/tools/export_perfetto.py

kernel 专属的 config / 调用封装放各自的 python 包（如 mega_moe）。
"""
from __future__ import annotations

import os

try:
    import tvm_ffi  # apache-tvm-ffi
except Exception as e:  # pragma: no cover
    tvm_ffi = None
    _IMPORT_ERR = e

_LIB_ENV = "MEGA_KERNEL_LIB"   # 指向编译产物 lib*_ffi.so


def load(lib_path: str | None = None):
    """加载编译好的 tvm-ffi 模块。lib_path 缺省读环境变量 MEGA_KERNEL_LIB。"""
    if tvm_ffi is None:
        raise ImportError(f"tvm_ffi not available: {_IMPORT_ERR}")
    path = lib_path or os.environ.get(_LIB_ENV)
    if not path:
        raise ValueError(f"set {_LIB_ENV} or pass lib_path to load()")
    return tvm_ffi.load_module(path)


def alloc_profiler_buffer(num_sms: int, num_groups: int, max_events: int):
    """分配 per-SM Perfetto 事件 buffer（int64 entries）。见 common/include/mega/profiler.cuh。"""
    import torch
    n = 1 + num_sms * num_groups * max_events          # +1 header
    return torch.zeros(n, dtype=torch.int64, device="cuda")


def dump_profiler_buffer(buf, path: str):
    """把 device buffer 拷回 host 并以 little-endian uint64 落盘，供 export_perfetto.py 解码。"""
    import numpy as np
    arr = buf.detach().to("cpu").view(-1).numpy().astype("<u8")
    arr.tofile(path)
    return path
