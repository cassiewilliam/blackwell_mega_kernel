// =============================================================================
// mega_moe.h —— public API
// -----------------------------------------------------------------------------
// Corresponds to DeepGEMM's `deep_gemm.fp8_fp4_mega_moe(...)` (mega/__init__.py
// 110-128), but drops the PyTorch tensors in favor of raw pointers + shape
// config, making it callable from pure C++.
//
// Calling convention (important):
//   * gate + top-k are not part of this project. The caller must provide the
//     pre-computed topk_idx / topk_weights, and pre-fill the inputs x / x_sf /
//     topk_* into the corresponding segments of the symmetric buffer (see
//     workspace.h).
//   * Weights must first be pre-processed by transform_weights_for_mega_moe(...):
//       L1 = interleave(gate/up) + transpose_sf_for_utccp
//       L2 = transpose_sf_for_utccp
//     (see src/weight_transform.cu, corresponding to mega/__init__.py 98-107)
//   * y is the BF16 output, [num_tokens, hidden].
// =============================================================================
#pragma once

#include <cstdint>

#include "mega_moe/shapes.h"

namespace mega_moe {

// A set of FP4 weights + their UE8M0 scale. For the layout, see the comments in
// weight_transform.cu.
struct Fp4Weights {
    const void* data;   // FP4 packed, [num_experts_per_rank, N, K/2]
    const void* scales; // UE8M0 SF (already transpose-for-UTCCP), [num_experts_per_rank, N, K/32]
};

// Handle to an already-rendezvoused symmetric buffer. For a single-GPU stub,
// peer_ptrs degenerates to {base}.
struct SymBufferView {
    void*  base;                 // base address of this rank's buffer
    void** peer_ptrs;            // array of device pointers to each rank's same-named buffer, length = num_ranks
    uint32_t rank;               // this rank's id
    uint32_t num_ranks;
    BufferLayout layout;         // per-segment offsets (obtained from compute_buffer_layout)
};

// -----------------------------------------------------------------------------
// Main entry point: launch the persistent Mega-MoE kernel on the stream.
//   y         : [num_tokens, hidden] BF16 output (device)
//   num_tokens: number of actually valid tokens for this rank (≤ num_max_tokens_per_rank)
//   l1 / l2   : pre-processed FP4 weights
//   buf       : symmetric buffer with inputs already filled in
//   cfg / tile: runtime mirror of the compile-time config (used to select the
//               already-instantiated template specialization)
//
// Returns cudaError_t (0 = success). For the template dispatch logic, see
// mega_moe_launch.cu.
// -----------------------------------------------------------------------------
int launch_mega_moe(void* y,
                    uint32_t num_tokens,
                    const Fp4Weights& l1,
                    const Fp4Weights& l2,
                    const SymBufferView& buf,
                    const MoEConfig& cfg,
                    const TileConfig& tile,
                    void* stream /* cudaStream_t */);

// The BLOCK_M actually used by the kernel (tokens must be aligned to it).
// Corresponds to _C.get_block_m_for_mega_moe.
uint32_t get_block_m(const MoEConfig& cfg, const TileConfig& tile);

}  // namespace mega_moe
