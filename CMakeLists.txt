cmake_minimum_required(VERSION 3.21)
project(VelriLib VERSION 0.4.0 LANGUAGES C)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(VELRI_LIB_TESTS_DEFAULT ON)
else()
    set(VELRI_LIB_TESTS_DEFAULT OFF)
endif()
option(VELRI_LIB_BUILD_TESTS "Build Velri-lib's correctness and contract test suites" ${VELRI_LIB_TESTS_DEFAULT})
option(VELRI_BUILD_BENCHMARKS "Build CPU and Vulkan roofline benchmark tools" OFF)
option(VELRI_CPU_MULTIVERSION "Build x86-64 AVX2 and AVX-512 AOT variants from the single Slang source" ON)
option(VELRI_SANITIZE "Enable address and undefined-behavior sanitizers" OFF)
option(VELRI_ENABLE_VULKAN "Build Slang AOT Vulkan backend (SPIR-V 1.6)" OFF)

if(VELRI_LIB_BUILD_TESTS)
    enable_testing()
endif()

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "Velri-lib requires Clang (including clang-cl) or GCC with C23 support")
endif()

set(VELRI_MSVC_FRONTEND OFF)
if(MSVC OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(VELRI_MSVC_FRONTEND ON)
endif()
set(VELRI_WINDOWS_MSVC_ABI OFF)
if(WIN32 AND (MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC"))
    set(VELRI_WINDOWS_MSVC_ABI ON)
endif()

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
if(VELRI_MSVC_FRONTEND)
    check_c_compiler_flag("/std:clatest" VELRI_HAS_C23_FLAG)
    if(NOT VELRI_HAS_C23_FLAG)
        message(FATAL_ERROR "Velri-lib requires C23 mode (/std:clatest); use Clang 18+ (clang-cl)")
    endif()
else()
    check_c_compiler_flag("-std=c23" VELRI_HAS_C23_FLAG)
    if(NOT VELRI_HAS_C23_FLAG)
        message(FATAL_ERROR "Velri-lib requires C23 mode; use Clang 18+ or GCC 14+")
    endif()
endif()

set(VELRI_X86_VARIANTS 0)
set(VELRI_BASE_FLAGS)
check_c_source_compiles("
#if !defined(__x86_64__) && !defined(_M_X64)
#error Not an x86-64 compiler target
#endif
int main(void) { return 0; }
" VELRI_TARGET_X86_64)
check_c_source_compiles("
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error Not an AArch64 compiler target
#endif
int main(void) { return 0; }
" VELRI_TARGET_AARCH64)

if(VELRI_TARGET_X86_64)
    if(NOT VELRI_MSVC_FRONTEND AND NOT ANDROID)
        set(VELRI_BASE_FLAGS -march=x86-64)
    endif()
    if(VELRI_CPU_MULTIVERSION)
        set(VELRI_X86_VARIANTS 1)
    endif()
endif()

function(velri_strict_target target)
    if(VELRI_MSVC_FRONTEND)
        target_compile_options(${target} PRIVATE /W4 /fp:precise /std:clatest)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        if(VELRI_SANITIZE)
            target_compile_options(${target} PRIVATE -fsanitize=address)
            target_link_options(${target} PUBLIC -fsanitize=address)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
        if(VELRI_WINDOWS_MSVC_ABI)
            target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        if(VELRI_SANITIZE)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PUBLIC -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

# Slang AOT build rules: sets VELRI_SLANG_CPU_OBJS and (optional) VELRI_VK_SHADER_HEADER.
include(cmake/slang.cmake)

# Single consolidated library `VelriLib::velri` (alias `VelriLib::core`)
if(WIN32)
    set(VELRI_PLATFORM_OS_SRC src/platform/os_win32.c)
else()
    find_package(Threads REQUIRED)
    set(VELRI_PLATFORM_OS_SRC src/platform/os_posix.c)
endif()

set(VELRI_SOURCES
    ${VELRI_PLATFORM_OS_SRC}
    src/platform/cpu_caps.c
    src/platform/pool.c
    src/core/tensor.c
    src/core/runtime_ctx.c
    src/ops/registry.c
    src/graph/graph.c
    src/blocks/builder.c
    src/blocks/norm.c
    src/blocks/embedding.c
    src/blocks/attention.c
    src/blocks/mlp.c
    src/blocks/lm_head.c
    src/layers/pre_norm.c
    src/plan/planner.c
    src/runtime/executor.c
    src/backend/cpu/cpu_backend.c
    ${VELRI_SLANG_CPU_OBJS}
)
if(VELRI_ENABLE_VULKAN)
    list(APPEND VELRI_SOURCES
        src/backend/vulkan/vk_device.c
        src/backend/vulkan/vk_memory.c
        src/backend/vulkan/vk_pipeline.c
        src/backend/vulkan/vk_decode_fusion.c
        src/backend/vulkan/vk_dispatch.c
        "${VELRI_VK_SHADER_HEADER}"
        "${VELRI_VK_ABI_HEADER}"
    )
endif()

add_library(velri STATIC ${VELRI_SOURCES})
add_library(VelriLib::velri ALIAS velri)
add_library(VelriLib::core ALIAS velri)
set_target_properties(velri PROPERTIES
    EXPORT_NAME velri
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(velri
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/platform"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ops"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/blocks"
)
if(VELRI_ENABLE_VULKAN)
    target_include_directories(velri PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/vulkan"
        "${VELRI_SLANG_GEN_DIR}"
        "${VELRI_VULKAN_INCLUDE_DIR}"
    )
    target_compile_definitions(velri PRIVATE VELRI_CORE_HAS_VULKAN=1)
endif()
target_compile_definitions(velri PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
target_compile_options(velri PRIVATE ${VELRI_BASE_FLAGS})
if(VELRI_MSVC_FRONTEND)
    target_compile_options(velri INTERFACE /std:clatest)
endif()
velri_strict_target(velri)
if(NOT WIN32)
    target_link_libraries(velri PUBLIC Threads::Threads PRIVATE ${CMAKE_DL_LIBS})
endif()
if(UNIX)
    target_link_libraries(velri PRIVATE m)
endif()

# Shared FP64 reference library (used by tests and optional benchmarks)
if(VELRI_LIB_BUILD_TESTS OR VELRI_BUILD_BENCHMARKS)
    add_library(velri_typed_reference STATIC tests/cpu/typed/reference.c)
    target_include_directories(velri_typed_reference PUBLIC tests/cpu/typed PRIVATE include)
    if(VELRI_MSVC_FRONTEND)
        target_compile_options(velri_typed_reference PRIVATE ${VELRI_BASE_FLAGS})
    else()
        target_compile_options(velri_typed_reference PRIVATE ${VELRI_BASE_FLAGS} -ffp-contract=off)
    endif()
    velri_strict_target(velri_typed_reference)
    if(UNIX)
        target_link_libraries(velri_typed_reference PUBLIC m)
    endif()
endif()

# Correctness & Contract Test Suites
if(VELRI_LIB_BUILD_TESTS)
    add_executable(test_kernels tests/test_kernels.c)
    target_compile_definitions(test_kernels PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
    if(VELRI_ENABLE_VULKAN)
        target_compile_definitions(test_kernels PRIVATE VELRI_CORE_HAS_VULKAN=1)
    endif()
    target_compile_options(test_kernels PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(test_kernels)
    target_link_libraries(test_kernels PRIVATE velri velri_typed_reference)
    add_test(NAME velri_kernels COMMAND test_kernels)

    add_executable(test_core_contract
        tests/core/test_core_contract.c
        tests/core/mock_async_backend.c
    )
    target_include_directories(test_core_contract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/platform")
    target_compile_options(test_core_contract PRIVATE ${VELRI_BASE_FLAGS})
    if(VELRI_ENABLE_VULKAN)
        target_compile_definitions(test_core_contract PRIVATE VELRI_CORE_HAS_VULKAN=1)
    endif()
    velri_strict_target(test_core_contract)
    target_link_libraries(test_core_contract PRIVATE velri velri_typed_reference)
    add_test(NAME core_contract COMMAND test_core_contract)
endif()

# Optional Benchmark Tools (excluded from production library & default test builds)
if(VELRI_BUILD_BENCHMARKS)
    add_executable(bench_slang_cpu tools/bench_slang_cpu.c)
    target_include_directories(bench_slang_cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/platform")
    target_compile_definitions(bench_slang_cpu PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
    target_compile_options(bench_slang_cpu PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(bench_slang_cpu)
    target_link_libraries(bench_slang_cpu PRIVATE velri velri_typed_reference)

    if(VELRI_ENABLE_VULKAN)
        add_executable(bench_vulkan tools/bench_vulkan.c "${VELRI_VK_CEILING_HEADER}")
        target_include_directories(bench_vulkan PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/platform"
            "${VELRI_SLANG_GEN_DIR}"
            "${VELRI_VULKAN_INCLUDE_DIR}"
        )
        target_compile_options(bench_vulkan PRIVATE ${VELRI_BASE_FLAGS})
        velri_strict_target(bench_vulkan)
        target_link_libraries(bench_vulkan PRIVATE velri velri_typed_reference)
    endif()
endif()

# Install CMake package exporting VelriLib::velri (+ VelriLib::core compatibility alias)
include(CMakePackageConfigHelpers)
install(TARGETS velri EXPORT VelriLibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT VelriLibTargets NAMESPACE VelriLib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VelriLib)
configure_package_config_file(cmake/VelriLibConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VelriLib)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VelriLib)
