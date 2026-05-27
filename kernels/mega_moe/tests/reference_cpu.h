// =============================================================================
// reference_cpu.h —— 五段 MoE 管线的 CPU 黄金参考
// -----------------------------------------------------------------------------
// 纯 FP32 host 实现，用于和 GPU kernel 对拍。刻意写得"傻而直白"：每一段一个函数，
// 直接照搬数学定义，不做任何量化/分块技巧——量化误差通过对拍容差吸收。
//
// 单 GPU stub（num_ranks == 1）语义：
//   Dispatch = 按 topk_idx 把 token gather 到对应 expert 的连续池；
//   Combine  = 按来源把 Linear2 输出 scatter 回去，并对 topk 维做加权求和。
// multi-GPU 时 dispatch/combine 跨 rank，但单 rank 内的数学完全一样。
// =============================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "mega_moe/shapes.h"

namespace mega_moe::ref {

// 行优先稠密矩阵的轻量视图（host FP32）。
struct Mat {
    std::vector<float> data;
    uint32_t rows = 0, cols = 0;
    float&       at(uint32_t r, uint32_t c)       { return data[(size_t)r * cols + c]; }
    const float& at(uint32_t r, uint32_t c) const { return data[(size_t)r * cols + c]; }
    Mat() = default;
    Mat(uint32_t r, uint32_t c) : data((size_t)r * c, 0.0f), rows(r), cols(c) {}
};

// 一次参考运行的全部输入（FP32，已"反量化"到真实数值域）。
struct RefInputs {
    Mat x;                                  // [num_tokens, H]
    std::vector<int32_t> topk_idx;          // [num_tokens * topk]，-1 = 该槽未路由
    std::vector<float>   topk_weights;      // [num_tokens * topk]
    std::vector<Mat>     l1_weights;        // per-expert，每个 [2*I, H]（gate‖up，未 interleave）
    std::vector<Mat>     l2_weights;        // per-expert，每个 [H, I]
};

// dispatch 后的 token 池 + 反查表，供逐段对拍。
struct DispatchResult {
    Mat pool_x;                             // [pool_n, H]，按 expert 连续排布
    std::vector<int32_t> tokens_per_expert; // [num_experts_per_rank]
    std::vector<int32_t> src_token;         // [pool_n] 每个池槽来自哪个 token
    std::vector<int32_t> src_expert;        // [pool_n] 落到哪个本地 expert
    std::vector<float>   src_weight;        // [pool_n] 对应 topk 权重
};

// ① Dispatch：把每个 token 的每个有效 topk 路由展开到对应 expert 的池中。
DispatchResult dispatch(const RefInputs& in, const MoEConfig& cfg);

// ② Linear1：pool_x @ W1ᵀ → [pool_n, 2*I]，gate‖up 并排。
Mat linear1(const DispatchResult& d, const RefInputs& in, const MoEConfig& cfg);

// ③ SwiGLU：out = silu(gate) * up * weight，逐池槽乘各自 topk 权重。clamp 可选。
Mat swiglu(const Mat& l1_out, const DispatchResult& d, const MoEConfig& cfg);

// ④ Linear2：swiglu_out @ W2ᵀ → [pool_n, H]。
Mat linear2(const Mat& act, const DispatchResult& d, const RefInputs& in, const MoEConfig& cfg);

// ⑤ Combine：按 src_token scatter 回 [num_tokens, H]，topk 维加权求和已含在池槽里。
Mat combine(const Mat& l2_out, const DispatchResult& d, uint32_t num_tokens, const MoEConfig& cfg);

// 端到端：跑完五段，返回 [num_tokens, H] 的 BF16-等价 FP32 结果。
Mat run_reference(const RefInputs& in, const MoEConfig& cfg, uint32_t num_tokens);

}  // namespace mega_moe::ref
