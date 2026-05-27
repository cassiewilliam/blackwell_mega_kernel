// =============================================================================
// mega_moe/events.h —— Perfetto event id definitions for mega_moe
// -----------------------------------------------------------------------------
// The generic probes live in common/include/mega/profiler.cuh; this file only
// defines the event semantics for the single mega_moe kernel. The group dimension
// is recommended to use warp-role, and event names must correspond one-to-one
// with export_perfetto.py's --events (or the default table).
// =============================================================================
#pragma once

#include <cstdint>

namespace mega_moe {

// Five stages + communication sub-events. The export script's default event
// table follows this order.
enum class Event : uint32_t {
    kDispatch = 0,   // ① NVLink pull
    kLinear1  = 1,   // ② FP8×FP4 GEMM
    kSwiGLU   = 2,   // ③ silu·up·weight + FP8 cast
    kLinear2  = 3,   // ④ FP8×FP4 GEMM
    kCombine  = 4,   // ⑤ NVLink push + topk reduce
    kTmaA     = 5,   // GEMM TMA-A load
    kTmaB     = 6,   // GEMM TMA-B load
    kMmaIssue = 7,   // UMMA issue
    kBarrier  = 8,   // cluster/grid sync wait
    kNumEvents = 9,
};

// warp-role = the profiler's group dimension.
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
