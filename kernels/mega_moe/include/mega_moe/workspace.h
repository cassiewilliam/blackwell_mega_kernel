// =============================================================================
// workspace.h —— host-side workspace / symmetric-buffer size computation
// -----------------------------------------------------------------------------
// Corresponds to DeepGEMM's `_C.get_symm_buffer_size_for_mega_moe(...)` (see
// mega/__init__.py 32-37). All of Mega-MoE's intermediate data lands in a single
// "symmetric buffer" — for a single-GPU stub it is just ordinary device memory;
// for multi-GPU it is NVLink symmetric memory (same address mapping on every
// rank), and dispatch/combine directly perform TMA pull/push on the remote
// rank's same-named buffer.
//
// This only does the host-side "compute sizes + slice views", and does not
// involve NVLink rendezvous (that part is in stage 2's symm_buffer.h). This
// header is host-only, pure C++, and can be compiled independently for tests.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

#include "mega_moe/shapes.h"

namespace mega_moe {

// Align v up to the nearest integer multiple of a.
constexpr uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// -----------------------------------------------------------------------------
// Byte offsets of each segment within the buffer (relative to the buffer base
// address). All segments are aligned to 256B to satisfy TMA.
// Segment meanings (corresponding to the SymmBuffer slices, mega/__init__.py 45-48):
//   x          : input tokens, FP8 E4M3                          [max_tokens, H]
//   x_sf       : UE8M0 scale of x, packed int                    [max_tokens, H/32 packed]
//   topk_idx   : routing target expert id, int32                 [max_tokens, topk]
//   topk_weights: routing weights, float                         [max_tokens, topk]
//   l1_acts    : token pool landing on this rank's experts after dispatch [pool_tokens, H] FP8
//   l1_acts_sf : scale of the above                              [pool_tokens, H/32]
//   l2_acts    : SwiGLU output (Linear2 input) FP8               [pool_tokens, I]
//   l2_acts_sf : scale of the above                              [pool_tokens, I/32]
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

    // pool capacity: max number of tokens this rank can take on = max_tokens * topk
    // (worst case: everything routes to this rank). The original further subdivides
    // by expert/wave; here we give a safe upper bound, sufficient for stage 1.
    uint64_t pool_tokens    = 0;
};

// FP8 E4M3 = 1 byte; FP4 packed = 0.5 byte; packed UE8M0 scale is estimated here
// as an upper bound at 1 byte/elem.
inline BufferLayout compute_buffer_layout(const MoEConfig& cfg, uint32_t block_m) {
    const uint64_t T   = align_up(cfg.num_max_tokens_per_rank, block_m);
    const uint64_t H   = cfg.hidden;
    const uint64_t I   = cfg.intermediate_hidden;
    const uint64_t TK  = cfg.num_topk;
    const uint64_t sfK = cfg.recipe.sf_block_k;       // 32
    const uint64_t pool = align_up(T * TK, block_m);  // upper bound of the token pool after dispatch

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
