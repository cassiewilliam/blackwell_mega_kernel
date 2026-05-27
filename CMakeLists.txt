# =============================================================================
# blackwell_mega_kernel —— SM100 (Blackwell) mega kernel 集合
# -----------------------------------------------------------------------------
# 顶层工程：把通用件（common/）与各 mega kernel（kernels/*/）组织在一起。
# 目前只有 mega_moe；后续新增的 mega kernel 在 kernels/ 下各占一个子目录，
# 复用 common/ 的 profiler、perfetto 导出、tvm-ffi 加载等基础设施。
# =============================================================================
cmake_minimum_required(VERSION 3.24)
project(blackwell_mega_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----- 全局选项（各 kernel 子工程共享读取）-----
option(MEGA_BUILD_KERNEL "Build CUDA SM100 kernels + GPU tests" OFF)
option(MEGA_BUILD_FFI "Build TVM FFI binding shared libraries" OFF)
option(MEGA_ENABLE_PROFILER "Enable per-SM Perfetto tracing probes (zero-cost when off)" OFF)
set(MEGA_CUTLASS_DIR "" CACHE PATH "Path to CUTLASS include/ (header-only)")
set(MEGA_TVM_FFI_DIR "" CACHE PATH "Path to apache-tvm-ffi package dir (include/ + lib/)")

# ----- 通用 INTERFACE 库：把 common/include 暴露给所有 kernel -----
add_library(mega_common INTERFACE)
target_include_directories(mega_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
if(MEGA_ENABLE_PROFILER)
    target_compile_definitions(mega_common INTERFACE MEGA_ENABLE_PROFILER)
endif()

enable_testing()

# ----- 各 kernel 子工程 -----
add_subdirectory(kernels/mega_moe)
# 未来：add_subdirectory(kernels/mega_ffn) 等
