# =============================================================================
# kernels/mega_moe —— SM100 FP8×FP4 Mega-MoE subproject
# -----------------------------------------------------------------------------
# Pulled in by the top-level CMakeLists via add_subdirectory, inheriting the
# MEGA_* global options and mega_common.
# Can also be built standalone: cmake -S kernels/mega_moe -B build.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# For standalone builds, fill in targets/options not defined by the top level
if(NOT TARGET mega_common)
    project(mega_moe LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    option(MEGA_BUILD_KERNEL "Build CUDA SM100 kernel + GPU tests" OFF)
    option(MEGA_BUILD_FFI "Build TVM FFI binding shared library" OFF)
    option(MEGA_ENABLE_PROFILER "Enable per-SM Perfetto tracing" OFF)
    set(MEGA_CUTLASS_DIR "" CACHE PATH "CUTLASS include/")
    set(MEGA_TVM_FFI_DIR "" CACHE PATH "apache-tvm-ffi dir")
    add_library(mega_common INTERFACE)
    target_include_directories(mega_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include)
    if(MEGA_ENABLE_PROFILER)
        target_compile_definitions(mega_common INTERFACE MEGA_ENABLE_PROFILER)
    endif()
    add_library(mega_vendor_deepgemm INTERFACE)
    target_include_directories(mega_vendor_deepgemm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../common/vendor)
    enable_testing()
endif()

set(MM_INC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----- host-only: CPU reference library (pure C++, always compilable) -----
add_library(mega_moe_ref STATIC tests/reference_cpu.cc)
target_include_directories(mega_moe_ref PUBLIC ${MM_INC} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(mega_moe_ref PUBLIC mega_common)

# host-only unit tests (no GPU required)
add_executable(mega_moe_test_layout tests/test_layout.cu)
set_source_files_properties(tests/test_layout.cu PROPERTIES LANGUAGE CXX)
target_link_libraries(mega_moe_test_layout PRIVATE mega_moe_ref)
add_test(NAME mega_moe_test_layout COMMAND mega_moe_test_layout)

# ----- CUDA kernel + GPU tests -----
if(MEGA_BUILD_KERNEL)
    enable_language(CUDA)
    if(NOT MEGA_CUTLASS_DIR)
        message(FATAL_ERROR "MEGA_BUILD_KERNEL=ON requires -DMEGA_CUTLASS_DIR=/path/to/cutlass/include")
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 100a)

    add_library(mega_moe STATIC
        src/mega_moe_sm100.cu
        src/mega_moe_launch.cu
        src/weight_transform.cu)
    target_include_directories(mega_moe PUBLIC ${MM_INC} ${MEGA_CUTLASS_DIR})
    target_link_libraries(mega_moe PUBLIC mega_common mega_vendor_deepgemm)
    target_compile_options(mega_moe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>)

    add_executable(mega_moe_test_single_gpu tests/test_single_gpu.cu)
    target_link_libraries(mega_moe_test_single_gpu PRIVATE mega_moe mega_moe_ref)
    add_test(NAME mega_moe_test_single_gpu COMMAND mega_moe_test_single_gpu)

    add_executable(mega_moe_bench bench/bench_mega_moe.cu)
    target_link_libraries(mega_moe_bench PRIVATE mega_moe mega_moe_ref)
endif()

# ----- TVM FFI binding shared library -----
if(MEGA_BUILD_FFI)
    if(NOT MEGA_BUILD_KERNEL)
        message(FATAL_ERROR "MEGA_BUILD_FFI=ON also requires -DMEGA_BUILD_KERNEL=ON")
    endif()
    if(NOT MEGA_TVM_FFI_DIR)
        message(FATAL_ERROR "MEGA_BUILD_FFI=ON requires -DMEGA_TVM_FFI_DIR=/path/to/tvm_ffi")
    endif()
    add_library(mega_moe_ffi SHARED bindings/mega_moe_ffi.cu)
    set_target_properties(mega_moe_ffi PROPERTIES PREFIX "lib")
    target_include_directories(mega_moe_ffi PRIVATE
        ${MM_INC} ${MEGA_CUTLASS_DIR} ${MEGA_TVM_FFI_DIR}/include)
    target_link_directories(mega_moe_ffi PRIVATE ${MEGA_TVM_FFI_DIR}/lib)
    target_link_libraries(mega_moe_ffi PRIVATE mega_moe mega_common tvm_ffi)
    target_compile_options(mega_moe_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>)
endif()
