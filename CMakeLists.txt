# =============================================================================
# blackwell_mega_kernel —— collection of SM100 (Blackwell) mega kernels
# -----------------------------------------------------------------------------
# Top-level project: organizes the common components (common/) together with
# each mega kernel (kernels/*/).
# Currently only mega_moe exists; future mega kernels each get their own
# subdirectory under kernels/, reusing common/ infrastructure such as the
# profiler, perfetto export, and tvm-ffi loading.
# =============================================================================
cmake_minimum_required(VERSION 3.24)
project(blackwell_mega_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----- Global options (shared/read by all kernel subprojects) -----
option(MEGA_BUILD_KERNEL "Build CUDA SM100 kernels + GPU tests" OFF)
option(MEGA_BUILD_FFI "Build TVM FFI binding shared libraries" OFF)
option(MEGA_ENABLE_PROFILER "Enable per-SM Perfetto tracing probes (zero-cost when off)" OFF)
set(MEGA_CUTLASS_DIR "" CACHE PATH "Path to CUTLASS include/ (header-only)")
set(MEGA_TVM_FFI_DIR "" CACHE PATH "Path to apache-tvm-ffi package dir (include/ + lib/)")

# ----- Common INTERFACE library: exposes common/include to all kernels -----
add_library(mega_common INTERFACE)
target_include_directories(mega_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
if(MEGA_ENABLE_PROFILER)
    target_compile_definitions(mega_common INTERFACE MEGA_ENABLE_PROFILER)
endif()

# ----- vendored DeepGEMM headers (only needed for CUDA builds; keeps the deep_gemm/ namespace) -----
add_library(mega_vendor_deepgemm INTERFACE)
target_include_directories(mega_vendor_deepgemm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common/vendor)

enable_testing()

# ----- Per-kernel subprojects -----
add_subdirectory(kernels/mega_moe)
# Future: add_subdirectory(kernels/mega_ffn), etc.
