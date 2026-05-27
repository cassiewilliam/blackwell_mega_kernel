#!/bin/bash
# build_ffi.sh — compile the TVM-FFI bridge .so inside the B200 container.
# Run from the repo root. Discovers torch / tvm_ffi / CUDA paths at build time.
set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
echo "repo=$REPO"

# --- discover toolchain paths (container: lmdeploy-fix-build) ---
TORCH=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
TVM_FFI=$(python3 -c "import tvm_ffi,os;print(os.path.dirname(tvm_ffi.__file__))")
ABI=$(python3 -c "import torch;print(int(torch.compiled_with_cxx11_abi()))")
CUDA=${CUDA_HOME:-/usr/local/cuda}
CUTLASS=${MEGA_CUTLASS_DIR:-/home/workcode/lmdeploy-trtllm-fused-moe/build/_deps/repo-cutlass-src/include}
NVLIB=$(python3 -c "import os,glob;print(os.path.dirname(glob.glob('/opt/py3/**/nvidia/cu13/lib/libnvrtc.so*',recursive=True)[0]))")
TVM_FFI_LIB=$(python3 -c "import os,glob;c=glob.glob('$TVM_FFI/**/libtvm_ffi*.so*',recursive=True);print(os.path.dirname(c[0]) if c else '')")
# Python.h needed because vendored csrc/apis/mega.hpp includes <pybind11/functional.h>
# (only for register_apis, which we don't call — but it still compiles).
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_path('include'))")

echo "torch=$TORCH abi=$ABI"
echo "tvm_ffi=$TVM_FFI lib=$TVM_FFI_LIB"
echo "cuda=$CUDA cutlass=$CUTLASS nvlib=$NVLIB"

OUT="$REPO/build_ffi"
mkdir -p "$OUT"

# --- merged HOST csrc tree (for the bridge g++) -------------------------------
# csrc uses RELATIVE includes (../jit/, ../../utils/, sm100.hpp), so the 3 editable
# MegaMoE host files (mega.hpp, launcher, heuristics/mega_moe.hpp) in src/csrc must
# sit in a tree alongside the shared csrc siblings. Build a symlink farm: MegaMoE
# files -> editable src/csrc; everything else -> common/vendor/csrc.
HOSTROOT="$OUT/host_root"
rm -rf "$HOSTROOT"; mkdir -p "$HOSTROOT/csrc/"{apis,jit_kernels/impls,jit_kernels/heuristics}
VC="$REPO/common/vendor/csrc"; SC="$REPO/kernels/mega_moe/src/csrc"
# shared dirs/files: symlink (their own relative includes resolve within vendor).
ln -sfn "$VC/jit"   "$HOSTROOT/csrc/jit"
ln -sfn "$VC/utils" "$HOSTROOT/csrc/utils"
ln -sfn "$VC/jit_kernels/impls/runtime_utils.hpp"            "$HOSTROOT/csrc/jit_kernels/impls/runtime_utils.hpp"
# editable MegaMoE host files: COPY (g++ canonicalizes symlinks, so a symlinked file's
# relative '../jit/' would resolve at the symlink TARGET dir, not here). Copy = real file
# in this merged tree -> relative includes hit the symlinked siblings above. Re-copied
# every build, so edits in src/csrc/ take effect; src/csrc is the source of truth.
cp "$SC/apis/mega.hpp"                                "$HOSTROOT/csrc/apis/mega.hpp"
cp "$SC/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp" "$HOSTROOT/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
cp "$SC/jit_kernels/heuristics/mega_moe.hpp"          "$HOSTROOT/csrc/jit_kernels/heuristics/mega_moe.hpp"
for h in sm100 sm90 common config runtime utils; do
  ln -sfn "$VC/jit_kernels/heuristics/$h.hpp" "$HOSTROOT/csrc/jit_kernels/heuristics/$h.hpp"
done

# Compile with g++ (host), NOT nvcc: the bridge is pure host code (the CUDA kernel
# is JIT-compiled by nvcc at runtime). Under nvcc, DG_IN_CUDA_COMPILATION would be
# defined and DeepGEMM's DG_UNIFIED_ASSERT would emit device `trap;` into the host
# object — DeepGEMM itself builds its host launcher with the host compiler for this
# reason. g++ leaves CUTLASS_HOST_DEVICE as plain inline and picks DG_HOST_ASSERT.
g++ -std=c++20 -O3 -fPIC -shared \
  -D_GLIBCXX_USE_CXX11_ABI=$ABI -DDG_TENSORMAP_COMPATIBLE=1 \
  -I"$HOSTROOT" \
  -I"$REPO/kernels/mega_moe/src" \
  -I"$REPO/common/vendor" \
  -I"$REPO/common/include" \
  -I"$CUTLASS" \
  -I"$TVM_FFI/include" \
  -I"$TORCH/include" \
  -I"$TORCH/include/torch/csrc/api/include" \
  -I"$PYINC" \
  -I"$CUDA/include" \
  -I"$CUDA/targets/x86_64-linux/include" \
  -I"$CUDA/targets/x86_64-linux/include/cccl" \
  -x c++ "$REPO/kernels/mega_moe/bindings/mega_moe_ffi.cu" \
  -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
  -L"$NVLIB" -lnvrtc -lcublasLt \
  ${TVM_FFI_LIB:+-L"$TVM_FFI_LIB" -ltvm_ffi} \
  -L"$CUDA/lib64" -lcudart -lcuda \
  -o "$OUT/libmega_moe_ffi.so"

echo "BUILD_OK: $OUT/libmega_moe_ffi.so"
ls -la "$OUT/libmega_moe_ffi.so"

# --- JIT include root ---------------------------------------------------------
# DeepGEMM's NVCC JIT compiles the kernel with a single -I{library_root}/include,
# which must contain deep_gemm/ + cute/ + cutlass/ (+ mega/ for profiler probes).
# Build a symlink farm and print the path; Python passes it to init().
JITROOT="$OUT/jit_root"
rm -rf "$JITROOT/include/deep_gemm"
mkdir -p "$JITROOT/include/deep_gemm/"{layout,scheduler,impls}
# deep_gemm/ is MERGED: shared infra -> common/vendor; the 3 MegaMoE files (kernel,
# layout, scheduler) -> the EDITABLE source in kernels/mega_moe/src. Edit those.
VDG="$REPO/common/vendor/deep_gemm"
SDG="$REPO/kernels/mega_moe/src/deep_gemm"
for d in comm common mma ptx; do ln -sfn "$VDG/$d" "$JITROOT/include/deep_gemm/$d"; done
ln -sfn "$VDG/layout/sym_buffer.cuh"                       "$JITROOT/include/deep_gemm/layout/sym_buffer.cuh"
ln -sfn "$SDG/layout/mega_moe.cuh"                         "$JITROOT/include/deep_gemm/layout/mega_moe.cuh"
ln -sfn "$SDG/scheduler/mega_moe.cuh"                      "$JITROOT/include/deep_gemm/scheduler/mega_moe.cuh"
ln -sfn "$SDG/impls/sm100_fp8_fp4_mega_moe.cuh"            "$JITROOT/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
ln -sfn "$REPO/common/include/mega" "$JITROOT/include/mega"
ln -sfn "$CUTLASS/cute"             "$JITROOT/include/cute"
ln -sfn "$CUTLASS/cutlass"          "$JITROOT/include/cutlass"
echo "JIT_ROOT=$JITROOT  (deep_gemm merged: mega files -> src/, shared -> vendor/)"
