// =============================================================================
// mega_moe/events.h —— mega_moe 的 Perfetto 事件 id 定义
// -----------------------------------------------------------------------------
// 通用探针在 common/include/mega/profiler.cuh，本文件只定义 mega_moe 这一个 kernel
// 的事件语义。group 维度建议用 warp-role，事件名须与 export_perfetto.py 的
// --events（或默认表）一一对应。
// =============================================================================
#pragma once

#include <cstdint>

namespace mega_moe {

// 五段 + 通信子事件。导出脚本默认事件表与此顺序一致。
enum class Event : uint32_t {
    kDispatch = 0,   // ① NVLink pull
    kLinear1  = 1,   // ② FP8×FP4 GEMM
    kSwiGLU   = 2,   // ③ silu·up·weight + FP8 cast
    kLinear2  = 3,   // ④ FP8×FP4 GEMM
    kCombine  = 4,   // ⑤ NVLink push + topk reduce
    kTmaA     = 5,   // GEMM TMA-A 载入
    kTmaB     = 6,   // GEMM TMA-B 载入
    kMmaIssue = 7,   // UMMA issue
    kBarrier  = 8,   // cluster/grid sync 等待
    kNumEvents = 9,
};

// warp-role = profiler 的 group 维度。
enum class Role : uint32_t {
    kDispatch = 0,
    kTmaA     = 1,
    kTmaB     = 2,
    kMma      = 3,
    kEpilogue = 4,
    kCombine  = 5,
    kNumRoles = 6,
};

}  // namespace mega_moe
