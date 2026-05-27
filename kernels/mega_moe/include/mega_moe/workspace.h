// =============================================================================
// workspace.h —— host 侧 workspace / symmetric-buffer 尺寸计算
// -----------------------------------------------------------------------------
// 对应 DeepGEMM 的 `_C.get_symm_buffer_size_for_mega_moe(...)`（见 mega/__init__.py 32-37）。
// Mega-MoE 的所有中间数据都落在一块"对称缓冲区"里——单 GPU stub 时它就是一块普通
// device 内存；multi-GPU 时它是 NVLink 对称内存（每个 rank 同地址映射），dispatch/combine
// 直接对远端 rank 的同名 buffer 做 TMA pull/push。
//
// 这里只做 host 侧的"算尺寸 + 切 view"，不涉及 NVLink rendezvous（那部分在阶段 2 的
// symm_buffer.h 里）。本头文件 host-only，纯 C++，可被测试单独编译。
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

#include "mega_moe/shapes.h"

namespace mega_moe {

// 把 v 向上对齐到 a 的整数倍。
constexpr uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// -----------------------------------------------------------------------------
// 缓冲区内各段的字节偏移（相对 buffer 基址）。所有段按 256B 对齐以满足 TMA。
// 段的含义（与 SymmBuffer slice 对应，mega/__init__.py 45-48）：
//   x          : 输入 token，FP8 E4M3                       [max_tokens, H]
//   x_sf       : x 的 UE8M0 scale，packed int               [max_tokens, H/32 packed]
//   topk_idx   : 路由目标 expert id，int32                  [max_tokens, topk]
//   topk_weights: 路由权重，float                            [max_tokens, topk]
//   l1_acts    : dispatch 后落到本 rank expert 的 token 池   [pool_tokens, H] FP8
//   l1_acts_sf : 上者的 scale                               [pool_tokens, H/32]
//   l2_acts    : SwiGLU 输出（Linear2 输入）FP8             [pool_tokens, I]
//   l2_acts_sf : 上者的 scale                               [pool_tokens, I/32]
// -----------------------------------------------------------------------------
struct BufferLayout {
    static constexpr uint64_t kAlign = 256;

    uint64_t off_x          = 0;
    uint64_t off_x_sf       = 0;
    uint64_t off_topk_idx   = 0;
    uint64_t off_topk_weights = 0;
    uint64_t off_l1_acts    = 0;
    uint64_t off_l1_acts_sf = 0;
    uint64_t off_l2_acts    = 0;
    uint64_t off_l2_acts_sf = 0;
    uint64_t total_bytes    = 0;

    // pool 容量：本 rank 最多承接的 token 数 = max_tokens * topk（最坏全路由到本 rank）
    // 实际原版会按 expert/wave 再细分，这里给安全上界，阶段 1 够用。
    uint64_t pool_tokens    = 0;
};

// FP8 E4M3 = 1 byte；FP4 packed = 0.5 byte；packed UE8M0 scale 这里按 1 byte/elem 估上界。
inline BufferLayout compute_buffer_layout(const MoEConfig& cfg, uint32_t block_m) {
    const uint64_t T   = align_up(cfg.num_max_tokens_per_rank, block_m);
    const uint64_t H   = cfg.hidden;
    const uint64_t I   = cfg.intermediate_hidden;
    const uint64_t TK  = cfg.num_topk;
    const uint64_t sfK = cfg.recipe.sf_block_k;       // 32
    const uint64_t pool = align_up(T * TK, block_m);  // dispatch 后的 token 池上界

    BufferLayout L;
    L.pool_tokens = pool;

    uint64_t o = 0;
    auto place = [&](uint64_t bytes) {
        uint64_t at = align_up(o, BufferLayout::kAlign);
        o = at + bytes;
        return at;
    };

    L.off_x            = place(T * H * 1);              // FP8
    L.off_x_sf         = place(T * (H / sfK) * 1);      // packed UE8M0
    L.off_topk_idx     = place(T * TK * sizeof(int32_t));
    L.off_topk_weights = place(T * TK * sizeof(float));
    L.off_l1_acts      = place(pool * H * 1);           // FP8
    L.off_l1_acts_sf   = place(pool * (H / sfK) * 1);
    L.off_l2_acts      = place(pool * I * 1);           // FP8
    L.off_l2_acts_sf   = place(pool * (I / sfK) * 1);

    L.total_bytes = align_up(o, BufferLayout::kAlign);
    return L;
}

}  // namespace mega_moe
