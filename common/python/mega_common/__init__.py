"""mega_common —— shared Python infrastructure for the kernels in blackwell_mega_kernel.

Only holds generic pieces unrelated to any specific kernel:
  * load(): load a compiled .so module via tvm_ffi
  * alloc_profiler_buffer(): allocate the per-SM Perfetto event buffer
  * dump_profiler_buffer(): write the buffer to disk, handed off to common/tools/export_perfetto.py

Kernel-specific config / call wrappers live in each kernel's own python package (e.g. mega_moe).
"""
from __future__ import annotations

import os

try:
    import tvm_ffi  # apache-tvm-ffi
except Exception as e:  # pragma: no cover
    tvm_ffi = None
    _IMPORT_ERR = e

_LIB_ENV = "MEGA_KERNEL_LIB"   # points to the build artifact lib*_ffi.so


def load(lib_path: str | None = None):
    """Load a compiled tvm-ffi module. lib_path defaults to the MEGA_KERNEL_LIB env var."""
    if tvm_ffi is None:
        raise ImportError(f"tvm_ffi not available: {_IMPORT_ERR}")
    path = lib_path or os.environ.get(_LIB_ENV)
    if not path:
        raise ValueError(f"set {_LIB_ENV} or pass lib_path to load()")
    return tvm_ffi.load_module(path)


def alloc_profiler_buffer(num_sms: int, num_groups: int, max_events: int):
    """Allocate the per-SM Perfetto event buffer (int64 entries). See common/include/mega/profiler.cuh."""
    import torch
    n = 1 + num_sms * num_groups * max_events          # +1 header
    return torch.zeros(n, dtype=torch.int64, device="cuda")


def dump_profiler_buffer(buf, path: str):
    """Copy the device buffer back to host and write it as little-endian uint64 to disk, for export_perfetto.py to decode."""
    import numpy as np
    arr = buf.detach().to("cpu").view(-1).numpy().astype("<u8")
    arr.tofile(path)
    return path
