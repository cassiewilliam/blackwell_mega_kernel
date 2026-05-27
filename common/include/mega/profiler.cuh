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

// per-(block,group) 写游标。closure 在 kernel 里按 group(=warp role) 构造一次。
struct Closure {
    Entry*   buffer;
    uint32_t base_tag;     // 已含 sm_id、block_id
    uint32_t slot;         // 1 + block*num_groups + group
    uint32_t stride;       // num_blocks * num_groups
    uint32_t cursor;       // 当前写到第几条
    bool     active;       // 只有指定线程写（通常 warp leader）

    __device__ __forceinline__ void emit(uint32_t event_idx, uint32_t type) {
        if (!active) return;
        Entry e;
        e.tag = base_tag | (event_idx << kIdxShift) | (type << kTypeShift);
        e.delta_time = read_globaltimer_lo();
        buffer[slot + (size_t)cursor * stride] = e;
        ++cursor;
    }
};

}  // namespace mega::prof

// kernel 入口处调用一次：写 header（仅 block0/thread0）并构造本 group 的 closure。
//   buf        : Entry* 全局 buffer
//   group_idx  : warp-role 编号 [0, num_groups)
//   num_groups : 总 group 数
//   write_pred : 只有该线程为 true 时本 group 才写（通常各 role 的 leader lane）
// NOTE: macro params are prefixed (p_*) to avoid colliding with Entry members
// like `num_groups` (otherwise the preprocessor rewrites `_h.num_groups`).
#define MEGA_PROFILER_INIT(p_buf, p_gi, p_ng, p_wp)                                     \
    ::mega::prof::Closure _mm_prof{};                                                   \
    do {                                                                                \
        const uint32_t _blk = blockIdx.x;                                               \
        const uint32_t _nbk = gridDim.x;                                                \
        if (_blk == 0 && threadIdx.x == 0) {                                            \
            ::mega::prof::Entry _h; _h.num_blocks = _nbk; _h.num_groups = (p_ng);       \
            (p_buf)[0] = _h;                                                            \
        }                                                                               \
        _mm_prof.buffer   = (p_buf);                                                    \
        _mm_prof.base_tag = ::mega::prof::encode_tag(                                   \
                                ::mega::prof::read_smid(), _blk, 0, 0);                 \
        _mm_prof.slot     = 1 + _blk * (p_ng) + (p_gi);                                 \
        _mm_prof.stride   = _nbk * (p_ng);                                              \
        _mm_prof.cursor   = 0;                                                          \
        _mm_prof.active   = (p_wp);                                                     \
    } while (0)

#define MEGA_PROFILE_BEGIN(ev)   _mm_prof.emit((uint32_t)(ev), ::mega::prof::kBegin)
#define MEGA_PROFILE_END(ev)     _mm_prof.emit((uint32_t)(ev), ::mega::prof::kEnd)
#define MEGA_PROFILE_INSTANT(ev) _mm_prof.emit((uint32_t)(ev), ::mega::prof::kInstant)

#else  // ---------------- 关闭：全部展开为空，零开销 ----------------

#define MEGA_PROFILER_INIT(p_buf, p_gi, p_ng, p_wp) ((void)0)
#define MEGA_PROFILE_BEGIN(ev)   ((void)0)
#define MEGA_PROFILE_END(ev)     ((void)0)
#define MEGA_PROFILE_INSTANT(ev) ((void)0)

#endif  // MEGA_ENABLE_PROFILER
