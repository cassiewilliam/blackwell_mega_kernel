// =============================================================================
// test_layout.cu —— host-only 单元测试：buffer 布局 + CPU 参考自洽性
// -----------------------------------------------------------------------------
// 阶段 1 不需要 GPU：用 .cu 扩展名只是为统一，CMake 里按 CXX 语言编译。
// 验证：(1) compute_buffer_layout 段不重叠且单调；(2) 五段 CPU 参考能跑通且
// dispatch→combine 的 token 守恒（每个 token 的输出 = 其 topk 专家贡献之和）。
// =============================================================================
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include "mega_moe/shapes.h"
#include "mega_moe/workspace.h"
#include "reference_cpu.h"

using namespace mega_moe;

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::printf("  [FAIL] %s\n", msg); ++g_failures; } } while (0)

// --- 测试 1：buffer 段偏移单调递增、对齐、总量自洽 ---
static void test_buffer_layout() {
    std::printf("test_buffer_layout\n");
    BufferLayout L = compute_buffer_layout(kSmokeSingleGpu, /*block_m=*/128);
    uint64_t offs[] = {L.off_x, L.off_x_sf, L.off_topk_idx, L.off_topk_weights,
                       L.off_l1_acts, L.off_l1_acts_sf, L.off_l2_acts, L.off_l2_acts_sf};
    for (int i = 0; i < 8; ++i) {
        CHECK(offs[i] % BufferLayout::kAlign == 0, "段未按 256B 对齐");
        if (i) CHECK(offs[i] > offs[i - 1], "段偏移未严格递增");
        CHECK(offs[i] < L.total_bytes, "段越界 total_bytes");
    }
    CHECK(L.pool_tokens >= kSmokeSingleGpu.num_max_tokens_per_rank, "pool 容量过小");
    std::printf("  total_bytes=%llu pool_tokens=%llu\n",
                (unsigned long long)L.total_bytes, (unsigned long long)L.pool_tokens);
}

// --- 测试 2：CPU 五段参考 token 守恒 ---
static void test_reference_pipeline() {
    std::printf("test_reference_pipeline\n");
    MoEConfig cfg = kSmokeSingleGpu;
    const uint32_t num_tokens = 64;
    const uint32_t H = cfg.hidden, I = cfg.intermediate_hidden;
    const uint32_t E = cfg.num_experts_per_rank(), TK = cfg.num_topk;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::uniform_int_distribution<int> exp(0, (int)cfg.num_experts - 1);

    ref::RefInputs in;
    in.x = ref::Mat(num_tokens, H);
    for (auto& v : in.x.data) v = uni(rng);
    in.topk_idx.resize((size_t)num_tokens * TK);
    in.topk_weights.resize((size_t)num_tokens * TK);
    for (uint32_t t = 0; t < num_tokens; ++t)
        for (uint32_t k = 0; k < TK; ++k) {
            in.topk_idx[t * TK + k] = exp(rng) % (int)E;  // 单 GPU：限定本地 expert
            in.topk_weights[t * TK + k] = std::abs(uni(rng));
        }
    for (uint32_t e = 0; e < E; ++e) {
        ref::Mat w1(cfg.l1_shape_n(), H); for (auto& v : w1.data) v = 0.01f * uni(rng);
        ref::Mat w2(H, I);                for (auto& v : w2.data) v = 0.01f * uni(rng);
        in.l1_weights.push_back(std::move(w1));
        in.l2_weights.push_back(std::move(w2));
    }

    ref::Mat y = ref::run_reference(in, cfg, num_tokens);
    CHECK(y.rows == num_tokens && y.cols == H, "输出形状错误");

    // 守恒性：手动重算一个 token 的输出，与端到端对拍
    ref::DispatchResult d = ref::dispatch(in, cfg);
    ref::Mat l1 = ref::linear1(d, in, cfg);
    ref::Mat act = ref::swiglu(l1, d, cfg);
    ref::Mat l2 = ref::linear2(act, d, in, cfg);
    ref::Mat y2 = ref::combine(l2, d, num_tokens, cfg);
    float max_diff = 0.f;
    for (size_t i = 0; i < y.data.size(); ++i)
        max_diff = std::fmax(max_diff, std::fabs(y.data[i] - y2.data[i]));
    CHECK(max_diff == 0.f, "端到端与分段重算不一致");
    std::printf("  pool_n=%u max_diff=%g\n", l1.rows, max_diff);
}

int main() {
    test_buffer_layout();
    test_reference_pipeline();
    if (g_failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
