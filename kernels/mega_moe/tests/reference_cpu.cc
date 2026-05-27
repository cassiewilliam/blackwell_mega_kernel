// =============================================================================
// reference_cpu.cc —— CPU reference implementation of the 5-stage MoE pipeline (see reference_cpu.h)
// =============================================================================
#include "reference_cpu.h"

#include <cassert>
#include <cmath>

namespace mega_moe::ref {

// silu(z) = z * sigmoid(z)
static inline float silu(float z) { return z / (1.0f + std::exp(-z)); }

// -----------------------------------------------------------------------------
// ① Dispatch
// -----------------------------------------------------------------------------
DispatchResult dispatch(const RefInputs& in, const MoEConfig& cfg) {
    const uint32_t num_tokens = in.x.rows;
    const uint32_t H  = cfg.hidden;
    const uint32_t TK = cfg.num_topk;
    const uint32_t E  = cfg.num_experts_per_rank();

    // Single-GPU stub: this rank owns expert ids [0, E). For multi-GPU, filter by rank offset here.
    DispatchResult d;
    d.tokens_per_expert.assign(E, 0);

    // First count how many tokens each local expert receives (used as start offset for contiguous layout).
    for (uint32_t t = 0; t < num_tokens; ++t)
        for (uint32_t k = 0; k < TK; ++k) {
            int32_t e = in.topk_idx[(size_t)t * TK + k];
            if (e >= 0 && (uint32_t)e < E) d.tokens_per_expert[e]++;
        }

    std::vector<int32_t> offset(E, 0);
    int32_t pool_n = 0;
    for (uint32_t e = 0; e < E; ++e) { offset[e] = pool_n; pool_n += d.tokens_per_expert[e]; }

    d.pool_x = Mat(pool_n, H);
    d.src_token.assign(pool_n, -1);
    d.src_expert.assign(pool_n, -1);
    d.src_weight.assign(pool_n, 0.0f);

    std::vector<int32_t> cursor = offset;
    for (uint32_t t = 0; t < num_tokens; ++t)
        for (uint32_t k = 0; k < TK; ++k) {
            int32_t e = in.topk_idx[(size_t)t * TK + k];
            if (e < 0 || (uint32_t)e >= E) continue;
            int32_t slot = cursor[e]++;
            for (uint32_t h = 0; h < H; ++h) d.pool_x.at(slot, h) = in.x.at(t, h);
            d.src_token[slot]  = (int32_t)t;
            d.src_expert[slot] = e;
            d.src_weight[slot] = in.topk_weights[(size_t)t * TK + k];
        }
    return d;
}

// -----------------------------------------------------------------------------
// ② Linear1: pool_x @ W1ᵀ, W1 shape [2*I, H], output [pool_n, 2*I]
// -----------------------------------------------------------------------------
Mat linear1(const DispatchResult& d, const RefInputs& in, const MoEConfig& cfg) {
    const uint32_t H = cfg.hidden;
    const uint32_t N = cfg.l1_shape_n();   // 2*I
    Mat out(d.pool_x.rows, N);
    for (uint32_t s = 0; s < d.pool_x.rows; ++s) {
        const Mat& W = in.l1_weights[d.src_expert[s]];   // [2*I, H]
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t h = 0; h < H; ++h) acc += d.pool_x.at(s, h) * W.at(n, h);
            out.at(s, n) = acc;
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// ③ SwiGLU: gate = out[:, :I], up = out[:, I:], act = silu(gate)*up*weight
// -----------------------------------------------------------------------------
Mat swiglu(const Mat& l1_out, const DispatchResult& d, const MoEConfig& cfg) {
    const uint32_t I = cfg.intermediate_hidden;
    const float clamp = cfg.activation_clamp;
    Mat act(l1_out.rows, I);
    for (uint32_t s = 0; s < l1_out.rows; ++s) {
        const float w = d.src_weight[s];
        for (uint32_t i = 0; i < I; ++i) {
            float gate = l1_out.at(s, i);
            float up   = l1_out.at(s, I + i);
            if (clamp > 0.0f) {   // Matches the original: first clamp gate/up to [-clamp, clamp]
                gate = std::fmax(-clamp, std::fmin(clamp, gate));
                up   = std::fmax(-clamp, std::fmin(clamp, up));
            }
            act.at(s, i) = silu(gate) * up * w;
        }
    }
    return act;
}

// -----------------------------------------------------------------------------
// ④ Linear2: act @ W2ᵀ, W2 shape [H, I], output [pool_n, H]
// -----------------------------------------------------------------------------
Mat linear2(const Mat& act, const DispatchResult& d, const RefInputs& in, const MoEConfig& cfg) {
    const uint32_t H = cfg.hidden;
    const uint32_t I = cfg.intermediate_hidden;
    Mat out(act.rows, H);
    for (uint32_t s = 0; s < act.rows; ++s) {
        const Mat& W = in.l2_weights[d.src_expert[s]];   // [H, I]
        for (uint32_t h = 0; h < H; ++h) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < I; ++i) acc += act.at(s, i) * W.at(h, i);
            out.at(s, h) = acc;
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// ⑤ Combine: scatter+accumulate by src_token (topk weighting is already folded into the pool slot's src_weight)
// -----------------------------------------------------------------------------
Mat combine(const Mat& l2_out, const DispatchResult& d, uint32_t num_tokens, const MoEConfig& cfg) {
    const uint32_t H = cfg.hidden;
    Mat y(num_tokens, H);   // zero-initialized
    for (uint32_t s = 0; s < l2_out.rows; ++s) {
        int32_t t = d.src_token[s];
        if (t < 0) continue;
        for (uint32_t h = 0; h < H; ++h) y.at(t, h) += l2_out.at(s, h);
    }
    return y;
}

// -----------------------------------------------------------------------------
// End-to-end
// -----------------------------------------------------------------------------
Mat run_reference(const RefInputs& in, const MoEConfig& cfg, uint32_t num_tokens) {
    DispatchResult d = dispatch(in, cfg);
    Mat l1 = linear1(d, in, cfg);
    Mat act = swiglu(l1, d, cfg);
    Mat l2 = linear2(act, d, in, cfg);
    return combine(l2, d, num_tokens, cfg);
}

}  // namespace mega_moe::ref
