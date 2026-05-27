// =============================================================================
// mega_moe_ffi.cu — TVM FFI bridge for DeepGEMM's fp8_fp4_mega_moe (keep-torch)
// -----------------------------------------------------------------------------
// We keep DeepGEMM's host launcher + NVCC JIT + torch tensors *unchanged*
// (vendored under common/vendor/csrc). This file is the only new host code: it
// replaces the 28-line pybind11 python_api.cpp with a TVM-FFI export.
//
// Flow: Python torch tensor --DLPack--> tvm::ffi::TensorView (here) --from_blob-->
// torch::Tensor --> deep_gemm::mega::fp8_fp4_mega_moe(...). torch::from_blob makes
// a non-owning view over the same device memory (zero copy); the launcher only
// reads data_ptr/shape/stride/dtype, so a view is sufficient.
//
// Tensors crossing the bridge are bf16 (y), int8 (FP4 packed weights), int32
// (weight SF), and int8/byte (sym_buffer) — FP8 activations live *inside* the
// sym_buffer and never cross here, so no FP8 dtype mapping is needed.
// =============================================================================
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

// DeepGEMM host API (vendored verbatim). Provides deep_gemm::mega::fp8_fp4_mega_moe
// and get_block_m_for_mega_moe. Pulls in the JIT + launcher + heuristics.
#include "csrc/apis/mega.hpp"

namespace ffi = tvm::ffi;
using ffi::TensorView;

namespace {

// One-time JIT setup — mirrors DeepGEMM's csrc/apis/runtime.hpp::init. Must be called
// before the first mega_moe(): tells the NVCC JIT where the headers (library_root/include
// must contain deep_gemm/ + cute/ + cutlass/ [+ mega/]) and CUDA toolkit live.
void Init(ffi::String library_root, ffi::String cuda_home) {
    const std::string root(library_root), cuda(cuda_home);
    deep_gemm::Compiler::prepare_init(root, cuda);
    deep_gemm::KernelRuntime::prepare_init(cuda);
    deep_gemm::IncludeParser::prepare_init(root);
}

// DLPack dtype -> torch scalar type (only the dtypes that cross this bridge).
at::ScalarType dl_to_torch(DLDataType dt) {
    if (dt.lanes != 1) TVM_FFI_THROW(TypeError) << "vector dtypes unsupported";
    switch (dt.code) {
        case kDLInt:
            if (dt.bits == 8)  return at::kChar;   // FP4 packed weights
            if (dt.bits == 32) return at::kInt;    // weight scale factors (UE8M0 packed)
            if (dt.bits == 64) return at::kLong;   // sym_buffer_ptrs
            break;
        case kDLUInt:
            if (dt.bits == 8)  return at::kByte;   // sym_buffer
            break;
        case kDLFloat:
            if (dt.bits == 16) return at::kHalf;
            if (dt.bits == 32) return at::kFloat;
            break;
        case kDLBfloat:
            if (dt.bits == 16) return at::kBFloat16;  // output y
            break;
        default: break;
    }
    TVM_FFI_THROW(TypeError) << "unsupported DLDataType code=" << int(dt.code)
                             << " bits=" << int(dt.bits);
}

// Non-owning torch view over a TensorView's device memory. Preserves strides for
// non-contiguous tensors (the transformed weight SF tensors are transposed views,
// which DeepGEMM's launcher reads via stride(-2)).
torch::Tensor to_torch(TensorView t) {
    const DLDevice dev = t.device();
    const int nd = t.ndim();
    std::vector<int64_t> sizes(nd);
    auto sh = t.shape();
    for (int i = 0; i < nd; ++i) sizes[i] = sh[i];
    auto opts = torch::TensorOptions()
                    .dtype(dl_to_torch(t.dtype()))
                    .device(torch::kCUDA, dev.device_id);
    if (t.IsContiguous())
        return torch::from_blob(t.data_ptr(), sizes, opts);
    std::vector<int64_t> strides(nd);
    auto st = t.strides();
    for (int i = 0; i < nd; ++i) strides[i] = st[i];
    return torch::from_blob(t.data_ptr(), sizes, strides, opts);
}

// Main entry — mirrors deep_gemm.fp8_fp4_mega_moe (recipe fixed to (1,1,32), swiglu).
void MegaMoE(TensorView y,
             TensorView l1_w, TensorView l1_sf,
             TensorView l2_w, TensorView l2_sf,
             TensorView sym_buffer,
             TensorView sym_buffer_ptrs,   // int64 [num_ranks]
             int64_t rank_idx,
             int64_t num_max_tokens_per_rank,
             int64_t num_experts, int64_t num_topk,
             double activation_clamp, bool fast_math) {
    const DLDevice dev = y.device();
    c10::cuda::CUDAGuard dev_guard(dev.device_id);
    // NOTE: DeepGEMM's launcher uses torch's current CUDA stream internally
    // (at::cuda::getCurrentCUDAStream). We deliberately do NOT override it with the
    // tvm-ffi stream here — the caller controls ordering via torch's current stream,
    // matching how deep_gemm.fp8_fp4_mega_moe itself runs.

    // sym_buffer_ptrs: copy int64 tensor into a std::vector (host side)
    auto ptrs_t = to_torch(sym_buffer_ptrs).to(torch::kCPU).contiguous();
    std::vector<int64_t> ptrs(ptrs_t.data_ptr<int64_t>(),
                              ptrs_t.data_ptr<int64_t>() + ptrs_t.numel());

    deep_gemm::mega::fp8_fp4_mega_moe(
        to_torch(y),
        std::make_tuple(to_torch(l1_w), to_torch(l1_sf)),
        std::make_tuple(to_torch(l2_w), to_torch(l2_sf)),
        std::optional<torch::Tensor>(std::nullopt),  // cumulative_local_expert_recv_stats
        to_torch(sym_buffer),
        ptrs, static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        std::make_tuple(1, 1, 32), "swiglu",
        std::optional<float>(static_cast<float>(activation_clamp)), fast_math);
}

// Token alignment helper (no tensors) — 2.5.0 renamed this from get_block_m_for_mega_moe.
int64_t TokenAlignment() {
    return deep_gemm::get_token_alignment_for_mega_moe();
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, Init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe, MegaMoE);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(token_alignment, TokenAlignment);
