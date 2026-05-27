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

# Compile with g++ (host), NOT nvcc: the bridge is pure host code (the CUDA kernel
# is JIT-compiled by nvcc at runtime). Under nvcc, DG_IN_CUDA_COMPILATION would be
# defined and DeepGEMM's DG_UNIFIED_ASSERT would emit device `trap;` into the host
# object — DeepGEMM itself builds its host launcher with the host compiler for this
# reason. g++ leaves CUTLASS_HOST_DEVICE as plain inline and picks DG_HOST_ASSERT.
g++ -std=c++20 -O3 -fPIC -shared \
  -D_GLIBCXX_USE_CXX11_ABI=$ABI -DDG_TENSORMAP_COMPATIBLE=1 \
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
