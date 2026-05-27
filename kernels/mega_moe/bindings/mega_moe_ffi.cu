// =============================================================================
// mega_moe_ffi.cu —— TVM FFI C++ 绑定层
// -----------------------------------------------------------------------------
// 用 TVM FFI 的 C++ 接口把 launch_mega_moe(...) 暴露成稳定 ABI 的函数，
// 可被 Python（torch tensor → DLPack → TensorView，零拷贝）或任何语言调用。
// 保留了 Python/JIT 工作流：Python 侧 tvm_ffi.load_module 本 .so 即可调用，
// 不再依赖 pybind11 / torch C++ 扩展（对比 DeepGEMM 的 csrc/python_api.cpp）。
//
// API 形态参考 tvm-ffi 官方 examples/kernel_library/scale_kernel.cu：
//   * 参数用 tvm::ffi::TensorView（非拥有视图）；.data_ptr() / .dtype() / .device()
//     / .ndim() / .size(i) / .numel()
//   * 设备守卫 ffi::CUDADeviceGuard，stream 经 TVMFFIEnvGetStream 取当前流
//   * 导出宏 TVM_FFI_DLL_EXPORT_TYPED_FUNC(symbol, func)
// =============================================================================
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "mega_moe/mega_moe.h"
#include "mega_moe/workspace.h"

namespace ffi = tvm::ffi;
using ffi::TensorView;

namespace {

// 轻量校验宏（tvm-ffi 风格）。
#define MM_CHECK_CUDA(t)                                                       \
    TVM_FFI_CHECK((t).device().device_type == kDLCUDA, ValueError)            \
        << #t " must be a CUDA tensor"
#define MM_CHECK_CONTIG(t)                                                     \
    TVM_FFI_CHECK((t).IsContiguous(), ValueError) << #t " must be contiguous"

cudaStream_t current_stream(DLDevice dev) {
    return static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));
}

// -----------------------------------------------------------------------------
// 主入口绑定。
//   y                 : [num_tokens, hidden] BF16 输出
//   l1_w / l1_sf      : 已 transform 的 L1 FP4 权重 + UE8M0 scale
//   l2_w / l2_sf      : 已 transform 的 L2 FP4 权重 + UE8M0 scale
//   sym_buffer        : 对称缓冲区（输入 x/x_sf/topk_* 已由调用者填好）
//   peer_ptrs         : int64 tensor，各 rank 同名 buffer 设备指针（单 GPU 时长度 1）
//   meta              : 形如 [num_tokens, num_max_tokens_per_rank, hidden, intermediate,
//                        num_experts, num_topk, num_ranks, rank, num_sms] 的 int64 tensor，
//                        把运行期标量打包传入，避免一长串 FFI 参数。
//   activation_clamp  : SwiGLU clamp（0 = 不 clamp）
//   fast_math         : 是否快速近似
//   profiler_buffer   : 可选，per-SM Perfetto 事件 buffer（int8/uint64）；空 tensor = 不开
// -----------------------------------------------------------------------------
void MegaMoE(TensorView y,
             TensorView l1_w, TensorView l1_sf,
             TensorView l2_w, TensorView l2_sf,
             TensorView sym_buffer,
             TensorView peer_ptrs,
             TensorView meta,
             double activation_clamp,
             bool fast_math,
             TensorView profiler_buffer) {
    MM_CHECK_CUDA(y); MM_CHECK_CUDA(sym_buffer);
    MM_CHECK_CONTIG(y); MM_CHECK_CONTIG(sym_buffer);
    TVM_FFI_CHECK(meta.dtype() == ffi::DataType::Int(64), ValueError)
        << "meta must be int64";

    const int64_t* m = static_cast<const int64_t*>(meta.data_ptr());
    mega_moe::MoEConfig cfg{};
    const uint32_t num_tokens = static_cast<uint32_t>(m[0]);
    cfg.num_max_tokens_per_rank = static_cast<uint32_t>(m[1]);
    cfg.hidden                  = static_cast<uint32_t>(m[2]);
    cfg.intermediate_hidden     = static_cast<uint32_t>(m[3]);
    cfg.num_experts             = static_cast<uint32_t>(m[4]);
    cfg.num_topk                = static_cast<uint32_t>(m[5]);
    cfg.num_ranks               = static_cast<uint32_t>(m[6]);
    const uint32_t rank         = static_cast<uint32_t>(m[7]);
    cfg.num_sms                 = static_cast<uint32_t>(m[8]);
    cfg.activation_clamp        = static_cast<float>(activation_clamp);
    cfg.fast_math               = fast_math;

    mega_moe::TileConfig tile{};
    const uint32_t block_m = mega_moe::get_block_m(cfg, tile);

    mega_moe::SymBufferView buf{};
    buf.base       = sym_buffer.data_ptr();
    buf.peer_ptrs  = reinterpret_cast<void**>(peer_ptrs.data_ptr());
    buf.rank       = rank;
    buf.num_ranks  = cfg.num_ranks;
    buf.layout     = mega_moe::compute_buffer_layout(cfg, block_m);

    mega_moe::Fp4Weights l1{ l1_w.data_ptr(), l1_sf.data_ptr() };
    mega_moe::Fp4Weights l2{ l2_w.data_ptr(), l2_sf.data_ptr() };

    // profiler buffer：非空则透传给 launch（阶段 2 接入 kernel）
    (void)profiler_buffer;  // TODO: launch_mega_moe 增加 profiler 入参后接上

    ffi::CUDADeviceGuard guard(y.device().device_id);
    cudaStream_t stream = current_stream(y.device());

    int rc = mega_moe::launch_mega_moe(y.data_ptr(), num_tokens, l1, l2, buf,
                                       cfg, tile, stream);
    TVM_FFI_CHECK(rc == 0, RuntimeError) << "launch_mega_moe failed, code=" << rc;
}

// host 侧算对称缓冲区字节数，给 Python 分配用。
int64_t SymmBufferBytes(TensorView meta) {
    const int64_t* m = static_cast<const int64_t*>(meta.data_ptr());
    mega_moe::MoEConfig cfg{};
    cfg.num_max_tokens_per_rank = static_cast<uint32_t>(m[1]);
    cfg.hidden                  = static_cast<uint32_t>(m[2]);
    cfg.intermediate_hidden     = static_cast<uint32_t>(m[3]);
    cfg.num_experts             = static_cast<uint32_t>(m[4]);
    cfg.num_topk                = static_cast<uint32_t>(m[5]);
    cfg.num_ranks               = static_cast<uint32_t>(m[6]);
    mega_moe::TileConfig tile{};
    const uint32_t block_m = mega_moe::get_block_m(cfg, tile);
    return static_cast<int64_t>(mega_moe::compute_buffer_layout(cfg, block_m).total_bytes);
}

}  // namespace

// 导出为 TVM FFI 符号；Python 侧用 mod.mega_moe / mod.symm_buffer_bytes 调用。
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe, MegaMoE);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(symm_buffer_bytes, SymmBufferBytes);
