# common/vendor —— 第三方 vendored 代码

## deep_gemm/

从 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)（MIT License, Copyright (c) 2025
DeepSeek，见 [deep_gemm/LICENSE](deep_gemm/LICENSE)）**原样搬运**的底层 SM100 头文件，
是 `sm100_fp8_fp4_mega_moe.cuh` 的完整传递 include 闭包（17 个文件 / ~3.7k LOC）。

这些是 Blackwell 的低层原语（PTX 封装、UMMA、TMA、SF 布局、grid/cluster barrier），
**重写收益低、风险高**，因此选择 vendor 而非重构——干净重构的精力放在 kernel 本身的
拆分、host launcher（去 JIT/NVRTC/torch）、tvm-ffi 绑定与 profiler 上。

保留 `deep_gemm/` 子目录命名空间，使各头文件内部的 `#include <deep_gemm/...>` 无需改动；
构建时把 `common/vendor` 加入 include path 即可。

| 子目录 | 内容 |
|---|---|
| `comm/barrier.cuh` | grid / cluster barrier（bit31-flip + ld.acquire） |
| `common/` | math / tma_copy / types / utils / exception / compile / cute_tie |
| `layout/` | mega_moe workspace 布局 + sym_buffer（NVLink 对称地址） |
| `mma/sm100.cuh` | UMMA descriptor + policy |
| `ptx/` | ld_st / tcgen05 / tma / utils 的内联 PTX |
| `scheduler/mega_moe.cuh` | wave-based expert 调度 |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 主 kernel（device 模板，待拆分重构） |

> 升级方式：从 DeepGEMM 对应 commit 重新 `cp` 这 17 个文件即可；不要在此目录手改逻辑。
> 本工程的改动（phase 拆分 / profiler anchor）会落到 `kernels/mega_moe/src/`，不污染 vendor。
