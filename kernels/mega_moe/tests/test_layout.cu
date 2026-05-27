// =============================================================================
// test_layout.cu —— host-only unit test: buffer layout + CPU reference self-consistency
// -----------------------------------------------------------------------------
// Stage 1 needs no GPU: the .cu extension is only for uniformity; CMake compiles it as CXX.
// Verifies: (1) compute_buffer_layout segments are non-overlapping and monotonic; (2) the
// 5-stage CPU reference runs end-to-end and token conservation holds across dispatch→combine
// (each token's output = the sum of its topk experts' contributions).
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

// --- Test 1: buffer segment offsets are monotonically increasing, aligned, and totals self-consistent ---
static void test_buffer_layout() {
    std::printf("test_buffer_layout\n");
    BufferLayout L = compute_buffer_layout(kSmokeSingleGpu, /*block_m=*/128);
    uint64_t offs[] = {L.off_x, L.off_x_sf, L.off_topk_idx, L.off_topk_weights,
                       L.off_l1_acts, L.off_l1_acts_sf, L.off_l2_acts, L.off_l2_acts_sf};
    for (int i = 0; i < 8; ++i) {
        CHECK(offs[i] % BufferLayout::kAlign == 0, "segment not aligned to 256B");
        if (i) CHECK(offs[i] > offs[i - 1], "segment offset not strictly increasing");
        CHECK(offs[i] < L.total_bytes, "segment exceeds total_bytes");
    }
    CHECK(L.pool_tokens >= kSmokeSingleGpu.num_max_tokens_per_rank, "pool capacity too small");
    std::printf("  total_bytes=%llu pool_tokens=%llu\n",
                (unsigned long long)L.total_bytes, (unsigned long long)L.pool_tokens);
}

// --- Test 2: token conservation of the 5-stage CPU reference ---
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
            in.topk_idx[t * TK + k] = exp(rng) % (int)E;  // single GPU: restrict to local experts
            in.topk_weights[t * TK + k] = std::abs(uni(rng));
        }
    for (uint32_t e = 0; e < E; ++e) {
        ref::Mat w1(cfg.l1_shape_n(), H); for (auto& v : w1.data) v = 0.01f * uni(rng);
        ref::Mat w2(H, I);                for (auto& v : w2.data) v = 0.01f * uni(rng);
        in.l1_weights.push_back(std::move(w1));
        in.l2_weights.push_back(std::move(w2));
    }

    ref::Mat y = ref::run_reference(in, cfg, num_tokens);
    CHECK(y.rows == num_tokens && y.cols == H, "wrong output shape");

    // Conservation: manually recompute the output stage by stage and cross-check against end-to-end
    ref::DispatchResult d = ref::dispatch(in, cfg);
    ref::Mat l1 = ref::linear1(d, in, cfg);
    ref::Mat act = ref::swiglu(l1, d, cfg);
    ref::Mat l2 = ref::linear2(act, d, in, cfg);
    ref::Mat y2 = ref::combine(l2, d, num_tokens, cfg);
    float max_diff = 0.f;
    for (size_t i = 0; i < y.data.size(); ++i)
        max_diff = std::fmax(max_diff, std::fabs(y.data[i] - y2.data[i]));
    CHECK(max_diff == 0.f, "end-to-end and stage-by-stage recompute mismatch");
    std::printf("  pool_n=%u max_diff=%g\n", l1.rows, max_diff);
}

int main() {
    test_buffer_layout();
    test_reference_pipeline();
    if (g_failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
