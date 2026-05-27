// =============================================================================
// reference_cpu.h —— CPU golden reference for the 5-stage MoE pipeline
// -----------------------------------------------------------------------------
// A pure FP32 host implementation, used to cross-check against the GPU kernel. Deliberately
// written "dumb and straightforward": one function per stage, copying the math definition
// directly, without any quantization/tiling tricks — quantization error is absorbed by the
// comparison tolerance.
//
// Single-GPU stub (num_ranks == 1) semantics:
//   Dispatch = gather tokens into the contiguous pool of their target expert per topk_idx;
//   Combine  = scatter the Linear2 output back by source, with weighted sum over the topk dim.
// For multi-GPU, dispatch/combine cross ranks, but the math within a single rank is identical.
// =============================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "mega_moe/shapes.h"

namespace mega_moe::ref {

// Lightweight view of a row-major dense matrix (host FP32).
struct Mat {
    std::vector<float> data;
    uint32_t rows = 0, cols = 0;
    float&       at(uint32_t r, uint32_t c)       { return data[(size_t)r * cols + c]; }
    const float& at(uint32_t r, uint32_t c) const { return data[(size_t)r * cols + c]; }
    Mat() = default;
    Mat(uint32_t r, uint32_t c) : data((size_t)r * c, 0.0f), rows(r), cols(c) {}
};

// All inputs for one reference run (FP32, already "dequantized" to the real value range).
struct RefInputs {
    Mat x;                                  // [num_tokens, H]
    std::vector<int32_t> topk_idx;          // [num_tokens * topk], -1 = slot not routed
    std::vector<float>   topk_weights;      // [num_tokens * topk]
    std::vector<Mat>     l1_weights;        // per-expert, each [2*I, H] (gate‖up, not interleaved)
    std::vector<Mat>     l2_weights;        // per-expert, each [H, I]
};

// Post-dispatch token pool + reverse-lookup tables, for stage-by-stage cross-checking.
struct DispatchResult {
    Mat pool_x;                             // [pool_n, H], laid out contiguously by expert
    std::vector<int32_t> tokens_per_expert; // [num_experts_per_rank]
    std::vector<int32_t> src_token;         // [pool_n] which token each pool slot comes from
    std::vector<int32_t> src_expert;        // [pool_n] which local expert it lands in
    std::vector<float>   src_weight;        // [pool_n] corresponding topk weight
};

// ① Dispatch: expand each valid topk route of every token into the pool of its target expert.
DispatchResult dispatch(const RefInputs& in, const MoEConfig& cfg);

// ② Linear1: pool_x @ W1ᵀ → [pool_n, 2*I], gate‖up side by side.
Mat linear1(const DispatchResult& d, const RefInputs& in, const MoEConfig& cfg);

// ③ SwiGLU: out = silu(gate) * up * weight, each pool slot multiplied by its own topk weight. clamp optional.
Mat swiglu(const Mat& l1_out, const DispatchResult& d, const MoEConfig& cfg);

// ④ Linear2: swiglu_out @ W2ᵀ → [pool_n, H].
Mat linear2(const Mat& act, const DispatchResult& d, const RefInputs& in, const MoEConfig& cfg);

// ⑤ Combine: scatter back to [num_tokens, H] by src_token; the weighted sum over the topk dim is already contained in the pool slots.
Mat combine(const Mat& l2_out, const DispatchResult& d, uint32_t num_tokens, const MoEConfig& cfg);

// End-to-end: run all five stages, returning the BF16-equivalent FP32 result of shape [num_tokens, H].
Mat run_reference(const RefInputs& in, const MoEConfig& cfg, uint32_t num_tokens);

}  // namespace mega_moe::ref
