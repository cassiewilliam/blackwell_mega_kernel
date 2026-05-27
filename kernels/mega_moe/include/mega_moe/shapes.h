// =============================================================================
// shapes.h —— 编译期 config / 形状 traits
// -----------------------------------------------------------------------------
// Mega-MoE 是一个 fully-templated kernel：所有形状、分块、线程数都是编译期常量，
// 这样编译器才能展开 TMA / UMMA 的 swizzle 与流水。本头文件把"一组自洽的模板参数"
// 打包成 traits 结构，避免在 launch 处手写一长串数字。
//
// 与 DeepGEMM 原版 `sm100_fp8_fp4_mega_moe_impl` 模板参数一一对应（见该文件 67-94 行）。
// =============================================================================
#pragma once

#include <cstdint>

namespace mega_moe {

// -----------------------------------------------------------------------------
// 量化 recipe：(block_m, block_n, block_k) 的 scale-factor 粒度。
// 默认 (1, 1, 32) = 沿 K 每 32 元素一个 UE8M0 scale（per-32），与原版一致。
// -----------------------------------------------------------------------------
struct Recipe {
    uint32_t sf_block_m = 1;
    uint32_t sf_block_n = 1;
    uint32_t sf_block_k = 32;
};

// -----------------------------------------------------------------------------
// MoE 规模参数（运行期可变的那部分以外的、决定 kernel 模板实例的部分）。
// -----------------------------------------------------------------------------
struct MoEConfig {
    // 【规模】
    uint32_t num_max_tokens_per_rank;   // 每 rank 最多 token 数（对齐到 BLOCK_M）
    uint32_t hidden;                    // H：输入/输出隐藏维
    uint32_t intermediate_hidden;       // I：FFN 中间维（gate/up 各占 I）
    uint32_t num_experts;               // 全局 expert 总数
    uint32_t num_topk;                  // 每 token 路由到的 expert 数

    // 【拓扑】
    uint32_t num_ranks;                 // NVLink 域内 rank 数（单 GPU stub 时 = 1）
    uint32_t num_sms;                   // 参与的 SM 数（persistent grid 大小）

    // 【数值】
    float activation_clamp;             // SwiGLU 前的 clamp（0 = 不 clamp）
    bool fast_math;                     // 是否走快速 silu/exp 近似

    Recipe recipe{};

    // —— 派生量 ——
    constexpr uint32_t num_experts_per_rank() const { return num_experts / num_ranks; }
    constexpr uint32_t l1_shape_n() const { return intermediate_hidden * 2; }  // gate‖up
    constexpr uint32_t l1_shape_k() const { return hidden; }
    constexpr uint32_t l2_shape_n() const { return hidden; }
    constexpr uint32_t l2_shape_k() const { return intermediate_hidden; }
};

// -----------------------------------------------------------------------------
// 分块 / 流水 / 线程 config（与 MoEConfig 解耦，便于调参 sweep）。
// 默认值来自 DeepGEMM 调优结果（perf_log 终态）。
// -----------------------------------------------------------------------------
struct TileConfig {
    uint32_t block_m = 128;             // token 分块（dispatch / GEMM 的 M 维）
    uint32_t block_n = 128;             // 权重输出分块（N 维）
    uint32_t block_k = 128;             // K 维分块（TMA swizzle 对齐 128B）
    uint32_t store_block_m = 64;        // epilogue 写回分块
    uint32_t num_stages = 4;            // K 维软件流水级数

    uint32_t num_experts_per_wave = 1;  // scheduler 每 wave 处理的 expert 数

    // warp 角色线程数（必须满足 kernel 的 static_assert，见原版 117-120 行）
    uint32_t num_dispatch_threads = 128;     // dispatch warps（% 128 == 0）
    uint32_t num_non_epilogue_threads = 128; // GEMM TMA+MMA（严格 == 128）
    uint32_t num_epilogue_threads = 128;     // epilogue + combine（% 128 == 0）
};

// -----------------------------------------------------------------------------
// Qwen3.5 默认 config（README 表格对应值）。num_ranks 默认 1 = 单 GPU stub。
// -----------------------------------------------------------------------------
constexpr MoEConfig kQwen35Default = MoEConfig{
    /*num_max_tokens_per_rank=*/ 8192,
    /*hidden=*/                  7168,
    /*intermediate_hidden=*/     3072,
    /*num_experts=*/             384,
    /*num_topk=*/                6,
    /*num_ranks=*/               1,      // 单 GPU stub；multi-GPU 时改 6
    /*num_sms=*/                 148,    // B200 SM 数；按实际设备调整
    /*activation_clamp=*/        0.0f,
    /*fast_math=*/               true,
    /*recipe=*/                  Recipe{},
};

// 单 GPU 冒烟测试用的小 config，便于 CPU 参考对拍。
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
