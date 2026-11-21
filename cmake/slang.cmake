# Khronos Slang AOT kernel compilation for CPU (mandatory) and Vulkan (optional).
# Pure build-time AOT via `slangc`: zero compiler/LLVM/libslang runtime dependency.
#
# Two CPU AOT lowering modes (`VELRI_SLANG_CPU_MODE`):
#   - LLVM_IR (default for compatible Clang): `slangc -target llvm-ir -O0` emits
#     target-neutral LLVM IR, stripped of host triple/datalayout, and CMake's own
#     Clang compiles it to `.o`/`.obj` for ANY OS & CPU architecture without hardcoding triples.
#   - OBJECT (automatic fallback for GCC or older Clang): `slangc -target host-object-code`
#     emits machine objects directly via bundled `slang-llvm`.

find_program(VELRI_SLANGC NAMES slangc
    HINTS "$ENV{HOME}/.local/bin" "$ENV{HOME}/.local/opt/slang/bin"
    NO_CMAKE_FIND_ROOT_PATH)
if(NOT VELRI_SLANGC)
    message(FATAL_ERROR
        "Velri-lib requires Khronos `slangc` in PATH (or ~/.local/bin) for AOT kernel compilation")
endif()

set(VELRI_SLANG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/slang")
set(VELRI_SLANG_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/slang")
file(MAKE_DIRECTORY "${VELRI_SLANG_GEN_DIR}")

set(VELRI_SLANG_COMMON_DEPS
    "${VELRI_SLANG_DIR}/common/numeric.slang"
    "${VELRI_SLANG_DIR}/common/activations.slang"
    "${VELRI_SLANG_DIR}/common/epilogue.slang"
)
set(VELRI_SLANG_CPU_DEPS
    ${VELRI_SLANG_COMMON_DEPS}
    "${VELRI_SLANG_DIR}/ops/norm.slang"
    "${VELRI_SLANG_DIR}/ops/rope.slang"
    "${VELRI_SLANG_DIR}/ops/elementwise.slang"
    "${VELRI_SLANG_DIR}/ops/linear_cpu.slang"
)
set(VELRI_SLANG_VK_DEPS
    ${VELRI_SLANG_COMMON_DEPS}
    "${VELRI_SLANG_DIR}/common/reductions.slang"
)

set(_velri_obj_ext "${CMAKE_C_OUTPUT_EXTENSION}")
if(NOT _velri_obj_ext)
    if(WIN32)
        set(_velri_obj_ext ".obj")
    else()
        set(_velri_obj_ext ".o")
    endif()
endif()

# Inherit cross-compilation toolchain flags cleanly without leaking host `-march=` into x86 fat variants.
set(_velri_clang_target_flags)
if(CMAKE_C_COMPILER_TARGET)
    list(APPEND _velri_clang_target_flags "--target=${CMAKE_C_COMPILER_TARGET}")
endif()
if(CMAKE_SYSROOT)
    list(APPEND _velri_clang_target_flags "--sysroot=${CMAKE_SYSROOT}")
endif()
if(NOT WIN32)
    list(APPEND _velri_clang_target_flags -fPIC)
endif()
set(_velri_raw_c_flags "${CMAKE_C_FLAGS}")
separate_arguments(_velri_c_flag_list NATIVE_COMMAND "${_velri_raw_c_flags}")
set(_velri_next_is_target OFF)
foreach(_f IN LISTS _velri_c_flag_list)
    if(_velri_next_is_target)
        list(APPEND _velri_clang_target_flags "${_f}")
        set(_velri_next_is_target OFF)
    elseif(_f STREQUAL "-target")
        list(APPEND _velri_clang_target_flags "${_f}")
        set(_velri_next_is_target ON)
    elseif(_f MATCHES "^(--target=|--sysroot=|-mabi=|-mfpu=|-mfloat-abi=)")
        list(APPEND _velri_clang_target_flags "${_f}")
    elseif(NOT VELRI_X86_VARIANTS AND _f MATCHES "^(-march=|-mcpu=|-mtune=)")
        list(APPEND _velri_clang_target_flags "${_f}")
    endif()
endforeach()

set(VELRI_SLANG_CPU_MODE "AUTO" CACHE STRING
    "Slang CPU AOT lowering mode: AUTO (LLVM_IR for compatible Clang, OBJECT for GCC/older Clang), LLVM_IR, or OBJECT")
set_property(CACHE VELRI_SLANG_CPU_MODE PROPERTY STRINGS AUTO LLVM_IR OBJECT)

set(_velri_use_llvm_ir OFF)
if(VELRI_SLANG_CPU_MODE STREQUAL "LLVM_IR" OR
   (VELRI_SLANG_CPU_MODE STREQUAL "AUTO" AND CMAKE_C_COMPILER_ID MATCHES "Clang"))
    set(_probe_dir "${VELRI_SLANG_GEN_DIR}/probe")
    file(MAKE_DIRECTORY "${_probe_dir}")
    file(WRITE "${_probe_dir}/probe.slang"
        "[DllExport] export __extern_cpp float velri_probe(float a, float b) { return a * b + 1.0f; }\n")
    execute_process(
        COMMAND "${VELRI_SLANGC}" "${_probe_dir}/probe.slang"
            -target llvm-ir -O0 -fp-mode fast -o "${_probe_dir}/probe.ll"
        RESULT_VARIABLE _probe_slang_rc
        OUTPUT_QUIET ERROR_QUIET
    )
    set(_probe_clang_rc 1)
    if(_probe_slang_rc EQUAL 0)
        execute_process(
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT_LL=${_probe_dir}/probe.ll"
                "-DOUTPUT_LL=${_probe_dir}/probe_clean.ll"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_ll_target.cmake"
            RESULT_VARIABLE _probe_strip_rc
        )
        if(_probe_strip_rc EQUAL 0)
            if(VELRI_MSVC_FRONTEND)
                set(_probe_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags}
                    /O2 /fp:fast -Wno-override-module -Qunused-arguments
                    /c "${_probe_dir}/probe_clean.ll" "/Fo${_probe_dir}/probe${_velri_obj_ext}")
            else()
                set(_probe_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags}
                    -O3 -ffast-math -fno-math-errno -Wno-override-module -Qunused-arguments
                    -c "${_probe_dir}/probe_clean.ll" -o "${_probe_dir}/probe${_velri_obj_ext}")
            endif()
            execute_process(
                COMMAND ${_probe_cmd}
                RESULT_VARIABLE _probe_clang_rc
                OUTPUT_QUIET ERROR_VARIABLE _probe_clang_err
            )
        endif()
    endif()
    if(_probe_clang_rc EQUAL 0)
        set(_velri_use_llvm_ir ON)
    elseif(VELRI_SLANG_CPU_MODE STREQUAL "LLVM_IR")
        message(FATAL_ERROR
            "VELRI_SLANG_CPU_MODE=LLVM_IR failed probe with ${CMAKE_C_COMPILER}:\n${_probe_clang_err}\n"
            "Ensure Clang's LLVM major version is >= slangc's bundled LLVM, or use AUTO/OBJECT.")
    endif()
endif()

if(_velri_use_llvm_ir)
    message(STATUS "Velri-lib Slang CPU AOT pipeline: target-neutral LLVM IR -> ${CMAKE_C_COMPILER}")
else()
    set(VELRI_SLANG_TARGET_TRIPLE "" CACHE STRING
        "Explicit LLVM target triple for slangc OBJECT mode (auto-detected when empty)")
    set(_velri_triple "${VELRI_SLANG_TARGET_TRIPLE}")
    if(NOT _velri_triple)
        if(WIN32)
            if(VELRI_TARGET_AARCH64)
                set(_velri_arch "aarch64")
            else()
                set(_velri_arch "x86_64")
            endif()
            if(VELRI_WINDOWS_MSVC_ABI)
                set(_velri_triple "${_velri_arch}-pc-windows-msvc")
            else()
                set(_velri_triple "${_velri_arch}-w64-windows-gnu")
            endif()
        elseif(CMAKE_C_COMPILER_TARGET)
            set(_velri_triple "${CMAKE_C_COMPILER_TARGET}")
        else()
            execute_process(
                COMMAND "${CMAKE_C_COMPILER}" -dumpmachine
                OUTPUT_VARIABLE _velri_dumpmachine
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _velri_dm_rc
            )
            if(_velri_dm_rc EQUAL 0 AND _velri_dumpmachine)
                set(_velri_triple "${_velri_dumpmachine}")
            elseif(VELRI_TARGET_AARCH64)
                set(_velri_triple "aarch64-linux-gnu")
            else()
                set(_velri_triple "x86_64-linux-gnu")
            endif()
        endif()
    endif()
    if(_velri_triple MATCHES "^([^-]+)-.*mingw32")
        set(_velri_triple "${CMAKE_MATCH_1}-w64-windows-gnu")
    endif()
    message(STATUS "Velri-lib Slang CPU AOT pipeline: slangc host-object-code (${_velri_triple})")
endif()

# ============================================================================
# 1. Slang AOT CPU Objects (ONE source compiled per ISA profile)
# ============================================================================
set(VELRI_SLANG_CPU_ENTRY "${VELRI_SLANG_DIR}/entry_cpu.slang")
set(VELRI_SLANG_CPU_OBJS)

function(velri_add_slang_cpu_variant isa mr nr_f32 nr_f64 llvm_features clang_isa_flags extra_defs)
    set(_obj "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}${_velri_obj_ext}")

    if(_velri_use_llvm_ir)
        set(_raw_ll "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}.raw.ll")
        set(_clean_ll "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}.ll")
        set(_slang_cmd "${VELRI_SLANGC}" "${VELRI_SLANG_CPU_ENTRY}"
            -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
            -target llvm-ir -O0 -fp-mode fast
            "-DVELRI_ISA=${isa}" "-DVELRI_MR=${mr}"
            "-DVELRI_NR_F32=${nr_f32}" "-DVELRI_NR_F64=${nr_f64}")
        foreach(_def IN LISTS extra_defs)
            list(APPEND _slang_cmd "-D${_def}")
        endforeach()
        list(APPEND _slang_cmd -o "${_raw_ll}")

        if(VELRI_MSVC_FRONTEND)
            set(_cc_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags} ${clang_isa_flags}
                /O2 /fp:fast -Wno-override-module -Qunused-arguments
                /c "${_clean_ll}" "/Fo${_obj}")
        else()
            set(_cc_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags} ${clang_isa_flags}
                -O3 -ffast-math -fno-math-errno -Wno-override-module -Qunused-arguments
                -c "${_clean_ll}" -o "${_obj}")
        endif()

        add_custom_command(
            OUTPUT "${_obj}"
            COMMAND ${_slang_cmd}
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT_LL=${_raw_ll}"
                "-DOUTPUT_LL=${_clean_ll}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_ll_target.cmake"
            COMMAND ${_cc_cmd}
            DEPENDS "${VELRI_SLANG_CPU_ENTRY}" ${VELRI_SLANG_CPU_DEPS}
                    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_ll_target.cmake"
            COMMENT "Slang AOT CPU (${isa}, LLVM IR -> Clang): entry_cpu.slang -> slang_cpu_${isa}${_velri_obj_ext}"
            VERBATIM
        )
    else()
        set(_cmd "${VELRI_SLANGC}" "${VELRI_SLANG_CPU_ENTRY}"
            -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
            -target host-object-code -O3 -fp-mode fast
            -llvm-target-triple "${_velri_triple}"
            "-DVELRI_ISA=${isa}" "-DVELRI_MR=${mr}"
            "-DVELRI_NR_F32=${nr_f32}" "-DVELRI_NR_F64=${nr_f64}")
        foreach(_def IN LISTS extra_defs)
            list(APPEND _cmd "-D${_def}")
        endforeach()
        if(llvm_features)
            list(APPEND _cmd -llvm-features "${llvm_features}")
        endif()
        list(APPEND _cmd -o "${_obj}")

        add_custom_command(
            OUTPUT "${_obj}"
            COMMAND ${_cmd}
            DEPENDS "${VELRI_SLANG_CPU_ENTRY}" ${VELRI_SLANG_CPU_DEPS}
            COMMENT "Slang AOT CPU (${isa}, ${_velri_triple}): entry_cpu.slang -> slang_cpu_${isa}${_velri_obj_ext}"
            VERBATIM
        )
    endif()

    set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set(VELRI_SLANG_CPU_OBJS ${VELRI_SLANG_CPU_OBJS} "${_obj}" PARENT_SCOPE)
endfunction()

set(_velri_base_mr 3)
set(_velri_base_defs "VELRI_SOFT_F16=1")
if(VELRI_TARGET_AARCH64)
    set(_velri_base_mr 4)
    set(_velri_base_defs "")
endif()
velri_add_slang_cpu_variant(base ${_velri_base_mr} 16 8 "" "${VELRI_BASE_FLAGS}" "${_velri_base_defs}")

if(VELRI_X86_VARIANTS)
    if(VELRI_MSVC_FRONTEND)
        set(_avx2_cc_flags /arch:AVX2)
        set(_avx512_cc_flags /arch:AVX512)
    else()
        set(_avx2_cc_flags ${VELRI_BASE_FLAGS} -mavx2 -mfma -mf16c)
        set(_avx512_cc_flags ${VELRI_BASE_FLAGS} -mavx2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c)
    endif()
    velri_add_slang_cpu_variant(avx2 3 32 16
        "+avx2,+fma,+f16c"
        "${_avx2_cc_flags}" "")
    velri_add_slang_cpu_variant(avx512 6 64 32
        "+avx2,+avx512f,+avx512bw,+avx512dq,+avx512vl,+fma,+f16c"
        "${_avx512_cc_flags}" "")
endif()

# ============================================================================
# 2. Slang AOT Vulkan SPIR-V 1.6 Shaders (optional)
# ============================================================================
function(velri_compile_vk_shaders out_header spv_dir header_path spec_list)
    file(MAKE_DIRECTORY "${spv_dir}")
    set(_entries)
    set(_spvs)
    foreach(_spec IN LISTS spec_list)
        string(REGEX REPLACE "\\|.*$" "" _src_path "${_spec}")
        string(REGEX REPLACE "^.*\\|" "" _entry "${_spec}")
        set(_spv "${spv_dir}/${_entry}.spv")
        set(_extra_caps)
        if(_entry STREQUAL "vk_gemm_coopmat_bf16" OR _entry STREQUAL "vk_gemm_coopmat_bf16_64" OR
           _entry STREQUAL "vk_ceiling_coopmat_bf16")
            list(APPEND _extra_caps -capability spvBFloat16KHR)
        endif()

        set(_val_cmd)
        set(_spv_comment "Slang AOT SPIR-V: ${_entry}.spv")
        if(VELRI_SPIRV_VAL)
            set(_val_cmd COMMAND "${VELRI_SPIRV_VAL}" --target-env vulkan1.3 "${_spv}")
            set(_spv_comment "Slang AOT SPIR-V + spirv-val: ${_entry}.spv")
        endif()
        add_custom_command(
            OUTPUT "${_spv}"
            COMMAND "${VELRI_SLANGC}" "${_src_path}"
                -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/vk"
                -target spirv -profile spirv_1_6 -Wno-41012 -fvk-use-c-layout -O3
                ${_extra_caps}
                -entry "${_entry}" -stage compute -o "${_spv}"
            ${_val_cmd}
            DEPENDS "${_src_path}" "${VELRI_SLANG_DIR}/vk/vk_common.slang" ${VELRI_SLANG_VK_DEPS}
            COMMENT "${_spv_comment}"
            VERBATIM
        )
        list(APPEND _entries "${_entry}")
        list(APPEND _spvs "${_spv}")
    endforeach()

    string(JOIN ";" _entry_list_arg ${_entries})
    add_custom_command(
        OUTPUT "${header_path}"
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT_DIR=${spv_dir}"
            "-DOUTPUT_HEADER=${header_path}"
            "-DENTRY_LIST=${_entry_list_arg}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        DEPENDS ${_spvs} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        COMMENT "Embedding SPIR-V shaders -> ${header_path}"
        VERBATIM
    )
    set(${out_header} "${header_path}" PARENT_SCOPE)
endfunction()

if(VELRI_ENABLE_VULKAN)
    find_path(VELRI_VULKAN_INCLUDE_DIR vulkan/vulkan.h
        HINTS "$ENV{HOME}/.local/opt/vulkan-headers/include" "$ENV{VULKAN_SDK}/Include"
              /usr/include /usr/local/include)
    if(NOT VELRI_VULKAN_INCLUDE_DIR)
        find_path(VELRI_VULKAN_INCLUDE_DIR vulkan/vulkan.h
            HINTS "$ENV{HOME}/.local/opt/vulkan-headers/include" "$ENV{VULKAN_SDK}/Include"
                  /usr/include /usr/local/include
            NO_CMAKE_FIND_ROOT_PATH)
    endif()
    if(NOT VELRI_VULKAN_INCLUDE_DIR)
        message(FATAL_ERROR "VELRI_ENABLE_VULKAN requires Vulkan-Headers (vulkan/vulkan.h)")
    endif()
    find_program(VELRI_SPIRV_VAL NAMES spirv-val NO_CMAKE_FIND_ROOT_PATH)

    set(_vk "${VELRI_SLANG_DIR}/vk")
    set(VELRI_VK_PROD_SPECS
        "${_vk}/norm.slang|vk_rmsnorm_f32"
        "${_vk}/norm.slang|vk_rmsnorm_bf16"
        "${_vk}/norm.slang|vk_rmsnorm_f16"
        "${_vk}/norm.slang|vk_fused_add_rmsnorm_f32"
        "${_vk}/norm.slang|vk_layernorm_f32"
        "${_vk}/rope.slang|vk_rope_f32"
        "${_vk}/rope.slang|vk_rope_bf16"
        "${_vk}/elementwise.slang|vk_activation_silu_f32"
        "${_vk}/elementwise.slang|vk_activation_gelu_f32"
        "${_vk}/elementwise.slang|vk_activation_relu2_f32"
        "${_vk}/elementwise.slang|vk_gated_silu_f32"
        "${_vk}/elementwise.slang|vk_gated_gelu_f32"
        "${_vk}/elementwise.slang|vk_vec_axpby_f32"
        "${_vk}/elementwise.slang|vk_vec_mul_f32"
        "${_vk}/elementwise.slang|vk_softmax_f32"
        "${_vk}/elementwise.slang|vk_fwht_f32"
        "${_vk}/elementwise.slang|vk_quantize_row_f32_i8"
        "${_vk}/gemv.slang|vk_gemv_f32"
        "${_vk}/gemv.slang|vk_gemv_f16"
        "${_vk}/gemv.slang|vk_gemv_bf16"
        "${_vk}/gemv.slang|vk_gemv_w8a8"
        "${_vk}/gemm_f32.slang|vk_gemm_f32"
        "${_vk}/gemm_f32.slang|vk_splitk_reduce"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_f16"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_bf16"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_w8a8"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_f16_64"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_bf16_64"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_w8a8_64"
    )
    velri_compile_vk_shaders(VELRI_VK_SHADER_HEADER
        "${VELRI_SLANG_GEN_DIR}/spv"
        "${VELRI_SLANG_GEN_DIR}/velri_vk_shaders.h"
        "${VELRI_VK_PROD_SPECS}"
    )

    if(VELRI_BUILD_BENCHMARKS)
        set(_ceil "${CMAKE_CURRENT_SOURCE_DIR}/tools/vk_ceilings.slang")
        set(VELRI_VK_CEIL_SPECS
            "${_ceil}|vk_ceiling_bw_f32x4"
            "${_ceil}|vk_ceiling_bw_read_f32x4"
            "${_ceil}|vk_ceiling_fma_f32"
            "${_ceil}|vk_ceiling_coopmat_f16"
            "${_ceil}|vk_ceiling_coopmat_bf16"
            "${_ceil}|vk_ceiling_coopmat_i8"
        )
        velri_compile_vk_shaders(VELRI_VK_CEILING_HEADER
            "${VELRI_SLANG_GEN_DIR}/spv_bench"
            "${VELRI_SLANG_GEN_DIR}/velri_vk_ceilings.h"
            "${VELRI_VK_CEIL_SPECS}"
        )
    endif()
endif()
