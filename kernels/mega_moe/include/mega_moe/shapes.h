// =============================================================================
// shapes.h —— compile-time config / shape traits
// -----------------------------------------------------------------------------
// Mega-MoE is a fully-templated kernel: all shapes, tilings, and thread counts
// are compile-time constants, so that the compiler can unroll the TMA / UMMA
// swizzle and pipeline. This header packs "one self-consistent set of template
// parameters" into a traits struct, avoiding hand-writing a long string of
// numbers at the launch site.
//
// Corresponds one-to-one with the template parameters of DeepGEMM's original
// `sm100_fp8_fp4_mega_moe_impl` (see lines 67-94 of that file).
// =============================================================================
#pragma once

#include <cstdint>

namespace mega_moe {

// -----------------------------------------------------------------------------
// Quantization recipe: scale-factor granularity of (block_m, block_n, block_k).
// Default (1, 1, 32) = one UE8M0 scale per 32 elements along K (per-32),
// matching the original.
// -----------------------------------------------------------------------------
struct Recipe {
    uint32_t sf_block_m = 1;
    uint32_t sf_block_n = 1;
    uint32_t sf_block_k = 32;
};

// -----------------------------------------------------------------------------
// MoE scale parameters (the part, beyond the runtime-variable portion, that
// determines the kernel template instance).
// -----------------------------------------------------------------------------
struct MoEConfig {
    // [Scale]
    uint32_t num_max_tokens_per_rank;   // max number of tokens per rank (aligned to BLOCK_M)
    uint32_t hidden;                    // H: input/output hidden dim
    uint32_t intermediate_hidden;       // I: FFN intermediate dim (gate/up each take I)
    uint32_t num_experts;               // total global expert count
    uint32_t num_topk;                  // number of experts each token is routed to

    // [Topology]
    uint32_t num_ranks;                 // number of ranks in the NVLink domain (= 1 for single-GPU stub)
    uint32_t num_sms;                   // number of participating SMs (persistent grid size)

    // [Numerics]
    float activation_clamp;             // clamp before SwiGLU (0 = no clamp)
    bool fast_math;                     // whether to use the fast silu/exp approximation

    Recipe recipe{};

    // —— derived quantities ——
    constexpr uint32_t num_experts_per_rank() const { return num_experts / num_ranks; }
    constexpr uint32_t l1_shape_n() const { return intermediate_hidden * 2; }  // gate‖up
    constexpr uint32_t l1_shape_k() const { return hidden; }
    constexpr uint32_t l2_shape_n() const { return hidden; }
    constexpr uint32_t l2_shape_k() const { return intermediate_hidden; }
};

// -----------------------------------------------------------------------------
// Tiling / pipeline / thread config (decoupled from MoEConfig for easy parameter
// sweeps). Default values come from the DeepGEMM tuning results (final state of
// perf_log).
// -----------------------------------------------------------------------------
struct TileConfig {
    uint32_t block_m = 128;             // token tiling (M dim of dispatch / GEMM)
    uint32_t block_n = 128;             // weight output tiling (N dim)
    uint32_t block_k = 128;             // K-dim tiling (TMA swizzle aligned to 128B)
    uint32_t store_block_m = 64;        // epilogue write-back tiling
    uint32_t num_stages = 4;            // number of K-dim software pipeline stages

    uint32_t num_experts_per_wave = 1;  // number of experts the scheduler processes per wave

    // warp-role thread counts (must satisfy the kernel's static_assert, see lines 117-120 of the original)
    uint32_t num_dispatch_threads = 128;     // dispatch warps (% 128 == 0)
    uint32_t num_non_epilogue_threads = 128; // GEMM TMA+MMA (strictly == 128)
    uint32_t num_epilogue_threads = 128;     // epilogue + combine (% 128 == 0)
};

// -----------------------------------------------------------------------------
// Qwen3.5 default config (values corresponding to the README table). num_ranks
// defaults to 1 = single-GPU stub.
// -----------------------------------------------------------------------------
constexpr MoEConfig kQwen35Default = MoEConfig{
    /*num_max_tokens_per_rank=*/ 8192,
    /*hidden=*/                  7168,
    /*intermediate_hidden=*/     3072,
    /*num_experts=*/             384,
    /*num_topk=*/                6,
    /*num_ranks=*/               1,      // single-GPU stub; change to 6 for multi-GPU
    /*num_sms=*/                 148,    // B200 SM count; adjust to the actual device
    /*activation_clamp=*/        0.0f,
    /*fast_math=*/               true,
    /*recipe=*/                  Recipe{},
};

// Small config for single-GPU smoke testing, convenient for cross-checking
// against the CPU reference.
constexpr MoEConfig kSmokeSingleGpu = MoEConfig{
    /*num_max_tokens_per_rank=*/ 256,
    /*hidden=*/                  512,
    /*intermediate_hidden=*/     256,
    /*num_experts=*/             8,
    /*num_topk=*/                2,
    /*num_ranks=*/               1,
    /*num_sms=*/                 16,
    /*activation_clamp=*/        0.0f,
    /*fast_math=*/               true,
    /*recipe=*/                  Recipe{},
};

}  // namespace mega_moe
