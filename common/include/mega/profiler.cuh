// =============================================================================
// mega/profiler.cuh —— per-SM Perfetto 时间线探针（device 侧，跨 kernel 共享）
// -----------------------------------------------------------------------------
// blackwell_mega_kernel 仓库的**通用**探针，所有 mega kernel（mega_moe / 未来的
// mega_ffn / mega_attention …）共用。仿照 FlashInfer 的 profiler.cuh（commit
// c802a05）：给每个 warp-role 区段打 begin/end 事件，事件携带 (timestamp,
// block_id, sm_id, event_id, event_type)，写进一块全局 buffer；跑完后用
// common/tools/export_perfetto.py 转成 Perfetto trace（每 SM 一条 track）。
//
// 设计要点：
//   * 全程由 MEGA_ENABLE_PROFILER 宏开关。**未定义时所有宏展开为空**，零开销。
//   * 事件记录 8 字节：union{ {num_blocks,num_groups} 头; {tag,delta_time} 事件 }
//     tag 位域： [1:0]=event_type  [11:2]=event_idx  [23:12]=block_id  [31:24]=sm_id
//   * sm_id  ← `mov.u32 %0, %smid;`；时间戳 ← `mov.u32 %0, %globaltimer_lo;`
//   * buffer 写偏移：每个 (block, group) 固定 slot，无 atomic：
//       slot   = 1 + block_id * num_groups + group_id   （+1 跳过 header）
//       stride = num_blocks * num_groups
//
// 每个 kernel 自定义自己的 event id 枚举（int 即可），并约定与 export_perfetto.py
// 的事件名称表一一对应。本头文件不绑定具体 kernel 的事件语义。
// =============================================================================
#pragma once

#include <cstdint>

namespace mega::prof {

struct Entry {
    union {
        struct { uint32_t num_blocks; uint32_t num_groups; };
        struct { uint32_t tag; uint32_t delta_time; };
        uint64_t raw;
    };
};

}  // namespace mega::prof

// -----------------------------------------------------------------------------
// 开关：默认关闭，零开销。打开方式：编译时 -DMEGA_ENABLE_PROFILER
// -----------------------------------------------------------------------------
#ifdef MEGA_ENABLE_PROFILER

namespace mega::prof {

enum EventType : uint32_t { kBegin = 0, kEnd = 1, kInstant = 2 };

static constexpr uint32_t kTypeShift  = 0;
static constexpr uint32_t kIdxShift   = 2;
static constexpr uint32_t kBlockShift = 12;
static constexpr uint32_t kSmShift    = 24;

__device__ __forceinline__ uint32_t read_smid() {
    uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}
__device__ __forceinline__ uint32_t read_globaltimer_lo() {
    uint32_t t; asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(t)); return t;
}
__device__ __forceinline__ uint32_t encode_tag(uint32_t sm, uint32_t block,
                                                uint32_t event_idx, uint32_t type) {
    return (sm << kSmShift) | (block << kBlockShift) |
           (event_idx << kIdxShift) | (type << kTypeShift);
}

// Stateless, group-aware write: re-derives blockIdx/gridDim/smid and takes the
// buffer pointer at EACH call. Holds NO state across calls — critical because
// warp-specialized kernels run `reg_reconfig`/setmaxnreg, which would corrupt any
// cached register state carried across the kernel body.
//   `group` = warp-role track [0, num_groups). Each (block, group) gets its own
//   begin/end slots. `active` selects the single writer thread (a role's leader).
//   Layout: slot = 1 + (block*num_groups + group) + cursor * (num_blocks*num_groups).
//   event_idx is stored in the tag (we set event_idx = group = role).
__device__ __forceinline__ void write_event(void* buf, bool active, uint32_t group,
                                             uint32_t num_groups, uint32_t cursor,
                                             uint32_t event_idx, uint32_t type) {
    if (buf == nullptr || !active) return;
    const uint32_t blk = blockIdx.x, nbk = gridDim.x;
    Entry e;
    e.tag = encode_tag(read_smid(), blk, event_idx, type);
    e.delta_time = read_globaltimer_lo();
    const uint32_t slot = 1 + (blk * num_groups + group);
    reinterpret_cast<Entry*>(buf)[slot + (size_t)cursor * (nbk * num_groups)] = e;
}

__device__ __forceinline__ void write_header(void* buf, uint32_t num_groups) {
    if (buf == nullptr || blockIdx.x != 0 || threadIdx.x != 0) return;
    Entry h; h.num_blocks = gridDim.x; h.num_groups = num_groups;
    reinterpret_cast<Entry*>(buf)[0] = h;
}

}  // namespace mega::prof

// Group-aware macros. `act` = is-this-thread-the-role-leader; `g` = group/role;
// `ng` = num_groups; `ev` = event id (we pass ev == g so the role is in the tag).
#define MEGA_PROFILER_INIT(p_buf, ng)             ::mega::prof::write_header((p_buf), (ng))
#define MEGA_PROFILE_BEGIN(p_buf, act, g, ng, ev) ::mega::prof::write_event((p_buf), (act), (g), (ng), 0u, (uint32_t)(ev), ::mega::prof::kBegin)
#define MEGA_PROFILE_END(p_buf, act, g, ng, ev)   ::mega::prof::write_event((p_buf), (act), (g), (ng), 1u, (uint32_t)(ev), ::mega::prof::kEnd)
// explicit cursor (for per-iteration probes inside loops; begin=2*iter, end=2*iter+1)
#define MEGA_PROFILE_BEGIN_AT(p_buf, act, g, ng, cur, ev) ::mega::prof::write_event((p_buf), (act), (g), (ng), (cur), (uint32_t)(ev), ::mega::prof::kBegin)
#define MEGA_PROFILE_END_AT(p_buf, act, g, ng, cur, ev)   ::mega::prof::write_event((p_buf), (act), (g), (ng), (cur), (uint32_t)(ev), ::mega::prof::kEnd)

#else  // ---------------- off: expand to nothing, zero cost ----------------

#define MEGA_PROFILER_INIT(p_buf, ng)             ((void)0)
#define MEGA_PROFILE_BEGIN(p_buf, act, g, ng, ev) ((void)0)
#define MEGA_PROFILE_END(p_buf, act, g, ng, ev)   ((void)0)
#define MEGA_PROFILE_BEGIN_AT(p_buf, act, g, ng, cur, ev) ((void)0)
#define MEGA_PROFILE_END_AT(p_buf, act, g, ng, cur, ev)   ((void)0)

#endif  // MEGA_ENABLE_PROFILER
