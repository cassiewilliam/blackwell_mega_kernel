// =============================================================================
// mega_moe.h —— 公开 API
// -----------------------------------------------------------------------------
// 对应 DeepGEMM 的 `deep_gemm.fp8_fp4_mega_moe(...)`（mega/__init__.py 110-128），
// 但去掉了 PyTorch tensor，改成裸指针 + 形状 config，纯 C++ 可调用。
//
// 调用约定（重要）：
//   * gate + top-k 不在本工程内。调用者必须提供算好的 topk_idx / topk_weights，
//     并把输入 x / x_sf / topk_* 预先填进 symmetric buffer 的对应段（见 workspace.h）。
//   * 权重必须先经 transform_weights_for_mega_moe(...) 预处理：
//       L1 = interleave(gate/up) + transpose_sf_for_utccp
//       L2 = transpose_sf_for_utccp
//     （见 src/weight_transform.cu，对应 mega/__init__.py 98-107）
//   * y 是 BF16 输出，[num_tokens, hidden]。
// =============================================================================
#pragma once

#include <cstdint>

#include "mega_moe/shapes.h"

namespace mega_moe {

// 一组 FP4 权重 + 其 UE8M0 scale。layout 见 weight_transform.cu 的注释。
struct Fp4Weights {
    const void* data;   // FP4 packed，[num_experts_per_rank, N, K/2]
    const void* scales; // UE8M0 SF（已 transpose-for-UTCCP），[num_experts_per_rank, N, K/32]
};

// 已 rendezvous 的对称缓冲区句柄。单 GPU stub 时 peer_ptrs 退化为 {base}。
struct SymBufferView {
    void*  base;                 // 本 rank buffer 基址
    void** peer_ptrs;            // 各 rank 同名 buffer 的设备指针数组，长度 = num_ranks
    uint32_t rank;               // 本 rank id
    uint32_t num_ranks;
    BufferLayout layout;         // 各段偏移（compute_buffer_layout 得到）
};

// -----------------------------------------------------------------------------
// 主入口：在 stream 上启动 persistent Mega-MoE kernel。
//   y         : [num_tokens, hidden] BF16 输出（device）
//   num_tokens: 本 rank 实际有效 token 数（≤ num_max_tokens_per_rank）
//   l1 / l2   : 已预处理的 FP4 权重
//   buf       : 已填好输入的对称缓冲区
//   cfg / tile: 编译期 config 的运行期镜像（用于选择已实例化的模板特化）
//
// 返回 cudaError_t（0 = 成功）。模板分发逻辑见 mega_moe_launch.cu。
// -----------------------------------------------------------------------------
int launch_mega_moe(void* y,
                    uint32_t num_tokens,
                    const Fp4Weights& l1,
                    const Fp4Weights& l2,
                    const SymBufferView& buf,
                    const MoEConfig& cfg,
                    const TileConfig& tile,
                    void* stream /* cudaStream_t */);

// kernel 实际使用的 BLOCK_M（token 须对齐到它）。对应 _C.get_block_m_for_mega_moe。
uint32_t get_block_m(const MoEConfig& cfg, const TileConfig& tile);

}  // namespace mega_moe
