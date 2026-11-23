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
    "${VELRI_SLANG_DIR}/common/key_iter.slang"
)
set(VELRI_SLANG_CPU_DEPS
    ${VELRI_SLANG_COMMON_DEPS}
    "${VELRI_SLANG_DIR}/ops/norm.slang"
    "${VELRI_SLANG_DIR}/ops/rope.slang"
    "${VELRI_SLANG_DIR}/ops/elementwise.slang"
    "${VELRI_SLANG_DIR}/ops/linear_cpu.slang"
    "${VELRI_SLANG_DIR}/ops/gather_copy.slang"
    "${VELRI_SLANG_DIR}/ops/attention.slang"
    "${VELRI_SLANG_DIR}/ops/batched_linear.slang"
)
set(VELRI_SLANG_VK_DEPS
    ${VELRI_SLANG_COMMON_DEPS}
    "${VELRI_SLANG_DIR}/common/reductions.slang"
    "${VELRI_SLANG_DIR}/common/reductions_wide.slang"
    "${VELRI_SLANG_DIR}/vk/attention_args.slang"
    "${VELRI_SLANG_DIR}/vk/flash_attention_coop_common.slang"
    "${VELRI_SLANG_DIR}/vk/flash_attention_half.slang"
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
# 1. Slang AOT CPU objects
# ============================================================================
set(VELRI_SLANG_CPU_ENTRY "${VELRI_SLANG_DIR}/entry_cpu.slang")
set(VELRI_SLANG_CPU_OBJS)

function(velri_add_slang_cpu_variant isa mr nr_f32 nr_f64 llvm_features clang_isa_flags)
    set(_obj "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}${_velri_obj_ext}")

    set(_slang_cmd "${VELRI_SLANGC}" "${VELRI_SLANG_CPU_ENTRY}"
        -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
        -fp-mode fast -DVELRI_ISA=${isa} -DVELRI_MR=${mr}
        -DVELRI_NR_F32=${nr_f32} -DVELRI_NR_F64=${nr_f64} ${ARGN})

    if(_velri_use_llvm_ir)
        set(_raw_ll "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}.raw.ll")
        set(_clean_ll "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}.ll")
        list(APPEND _slang_cmd -target llvm-ir -O0 -o "${_raw_ll}")

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
            COMMENT "Slang AOT CPU (${isa}, LLVM IR -> Clang): slang_cpu_${isa}${_velri_obj_ext}"
            VERBATIM
        )
    else()
        list(APPEND _slang_cmd -target host-object-code -O3 -llvm-target-triple "${_velri_triple}")
        if(llvm_features)
            list(APPEND _slang_cmd -llvm-features "${llvm_features}")
        endif()
        list(APPEND _slang_cmd -o "${_obj}")

        add_custom_command(
            OUTPUT "${_obj}"
            COMMAND ${_slang_cmd}
            DEPENDS "${VELRI_SLANG_CPU_ENTRY}" ${VELRI_SLANG_CPU_DEPS}
            COMMENT "Slang AOT CPU (${isa}, ${_velri_triple}): slang_cpu_${isa}${_velri_obj_ext}"
            VERBATIM
        )
    endif()

    set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set(VELRI_SLANG_CPU_OBJS ${VELRI_SLANG_CPU_OBJS} "${_obj}" PARENT_SCOPE)
endfunction()

set(_velri_base_mr 3)
set(_velri_base_defs "-DVELRI_SOFT_F16=1")
if(VELRI_TARGET_AARCH64)
    set(_velri_base_mr 4)
    set(_velri_base_defs "")
endif()

velri_add_slang_cpu_variant(base ${_velri_base_mr} 16 8 "" "${VELRI_BASE_FLAGS}" ${_velri_base_defs})

# Check the emitted byte dot and its C ABI before using a native compiler.
function(velri_probe_slang_dot result cc input)
    set(${result} FALSE PARENT_SCOPE)
    get_filename_component(_dir "${input}" DIRECTORY)
    get_filename_component(_stem "${input}" NAME_WE)
    set(_obj "${_dir}/${_stem}.o")
    file(REMOVE "${_obj}" "${_dir}/${_stem}.s")
    execute_process(COMMAND "${cc}" ${ARGN} -save-temps=obj -c "${input}" -o "${_obj}"
        WORKING_DIRECTORY "${_dir}" RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
    if(NOT _rc EQUAL 0 OR NOT EXISTS "${_dir}/${_stem}.s")
        return()
    endif()
    file(STRINGS "${_dir}/${_stem}.s" _dot REGEX "vpdpbusd[ \t].*zmm" LIMIT_COUNT 1)
    if(_dot)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        try_compile(_links "${_dir}/link-${_stem}" "${_dir}/main.c" LINK_LIBRARIES "${_obj}")
        set(${result} ${_links} PARENT_SCOPE)
    endif()
endfunction()

function(velri_add_slang_cpu_dot)
    set(_probe "${VELRI_SLANG_GEN_DIR}/probe/dot")
    file(MAKE_DIRECTORY "${_probe}")
    file(WRITE "${_probe}/dot.slang" [=[
[DllExport] export __extern_cpp uint velri_probe_dot(Ptr<uint8_t, Access.Read> x,
    Ptr<int8_t, Access.Read> w, uint k) {
    uint sum = 0;
    for (uint i = 0; i < k; ++i) sum += uint(int(x[i]) * int(w[i]));
    return sum;
}
]=])
    file(WRITE "${_probe}/main.c" [=[
#include <stdint.h>
extern uint32_t velri_probe_dot(const uint8_t *, const int8_t *, uint32_t);
int main(void) { return (int)velri_probe_dot(0, 0, 0); }
]=])
    set(_flags -O3 -fPIC ${_avx512_cc_flags} -mavx512vnni -mprefer-vector-width=512)
    set(_source "${VELRI_SLANG_DIR}/ops/w8a8_dot.slang")
    set(_base "${VELRI_SLANG_GEN_DIR}/slang_cpu_dot")
    set(_deps "${_source}" "${VELRI_SLANG_DIR}/common/epilogue.slang" "${VELRI_SLANGC}")
    set(_strip "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_ll_target.cmake")

    # A compiler with byte partial-reduction support can keep every GEMM
    # accumulator in a byte-dot instruction. Check its actual lowering.
    set(_clang_name clang)
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        set(_clang_name "${CMAKE_C_COMPILER}")
    endif()
    set(VELRI_SLANG_DOT_CLANG "${_clang_name}" CACHE STRING "Native Clang name or path for Slang W8A8 LLVM IR")
    find_program(_dot_clang NAMES "${VELRI_SLANG_DOT_CLANG}" NO_CACHE)
    if(_dot_clang)
        set(_llvm_flags ${_flags} -Wno-override-module -mllvm -force-vector-width=64
            -mllvm -enable-epilogue-vectorization -mllvm -epilogue-vectorization-force-VF=8)
        execute_process(COMMAND "${VELRI_SLANGC}" "${_probe}/dot.slang" -target llvm-ir -O0
            -o "${_probe}/raw.ll" RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        if(_rc EQUAL 0)
            execute_process(COMMAND "${CMAKE_COMMAND}" "-DINPUT_LL=${_probe}/raw.ll"
                "-DOUTPUT_LL=${_probe}/llvm-dot.ll" -P "${_strip}"
                RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        endif()
        if(_rc EQUAL 0)
            velri_probe_slang_dot(_llvm "${_dot_clang}" "${_probe}/llvm-dot.ll" ${_llvm_flags})
        endif()
    endif()

    if(_llvm)
        add_custom_command(OUTPUT "${_base}.o" BYPRODUCTS "${_base}.raw.ll" "${_base}.ll"
            COMMAND "${VELRI_SLANGC}" "${_source}" -target llvm-ir -O0
                -I "${VELRI_SLANG_DIR}/common" -o "${_base}.raw.ll"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT_LL=${_base}.raw.ll" "-DOUTPUT_LL=${_base}.ll" -P "${_strip}"
            COMMAND "${_dot_clang}" ${_llvm_flags} -c "${_base}.ll" -o "${_base}.o"
            DEPENDS ${_deps} "${_strip}" "${_dot_clang}" VERBATIM
            COMMENT "Slang AOT CPU (W8A8 byte dot, LLVM IR -> Clang)")
        message(STATUS "Velri-lib Slang W8A8 byte-dot AOT: LLVM IR -> ${_dot_clang}")
    else()
        set(_gxx_names g++ g++-16 g++-15 g++-14)
        if(DEFINED VELRI_SLANG_GXX)
            set(_gxx_names "${VELRI_SLANG_GXX}")
        endif()
        find_program(_dot_gxx NAMES ${_gxx_names} NO_CACHE)
        if(NOT _dot_gxx)
            return()
        endif()
        list(APPEND _flags -std=c++17 -fno-exceptions -fno-rtti -fno-strict-aliasing)
        execute_process(COMMAND "${VELRI_SLANGC}" "${_probe}/dot.slang" -target cpp -O3
            -o "${_probe}/dot.cpp" RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        if(_rc EQUAL 0)
            velri_probe_slang_dot(_gxx "${_dot_gxx}" "${_probe}/dot.cpp" ${_flags})
        endif()
        if(NOT _gxx)
            return()
        endif()
        add_custom_command(OUTPUT "${_base}.cpp"
            COMMAND "${VELRI_SLANGC}" "${_source}" -target cpp -O3
                -I "${VELRI_SLANG_DIR}/common" -o "${_base}.cpp"
            DEPENDS ${_deps} VERBATIM COMMENT "Slang AOT CPU (W8A8 byte dot -> C++)")
        add_custom_command(OUTPUT "${_base}.o"
            COMMAND "${_dot_gxx}" ${_flags} -MD -MF "${_base}.o.d" -c "${_base}.cpp" -o "${_base}.o"
            DEPENDS "${_base}.cpp" "${_dot_gxx}" DEPFILE "${_base}.o.d"
            VERBATIM COMMENT "Slang AOT CPU (W8A8 byte dot, G++)")
        message(STATUS "Velri-lib Slang W8A8 byte-dot AOT: C++ -> ${_dot_gxx}")
    endif()
    set_source_files_properties("${_base}.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set(VELRI_SLANG_CPU_OBJS ${VELRI_SLANG_CPU_OBJS} "${_base}.o" PARENT_SCOPE)
    set(VELRI_HAS_SLANG_CPU_DOT 1 PARENT_SCOPE)
endfunction()

set(VELRI_HAS_SLANG_CPU_DOT 0)
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
        "${_avx2_cc_flags}")
    set(_avx512_features "+avx2,+avx512f,+avx512bw,+avx512dq,+avx512vl,+fma,+f16c")
    velri_add_slang_cpu_variant(avx512 6 64 32 "${_avx512_features}"
        "${_avx512_cc_flags}")
    if(VELRI_SLANG_CPU_DOT AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_CROSSCOMPILING
       AND NOT CMAKE_SYSROOT AND NOT CMAKE_C_COMPILER_TARGET AND NOT VELRI_SLANG_TARGET_TRIPLE
       AND NOT _velri_raw_c_flags MATCHES "(^| )(-target |--target=|--sysroot=)")
        velri_add_slang_cpu_dot()
        if(NOT VELRI_HAS_SLANG_CPU_DOT)
            message(STATUS "Velri-lib Slang W8A8: using standard AOT (byte-dot probe unavailable)")
        endif()
    endif()
endif()

# ============================================================================
# 2. Slang AOT Vulkan SPIR-V 1.6 Shaders (optional)
# ============================================================================
function(velri_compile_vk_shaders out_header spv_dir header_path spec_list abi_header)
    file(MAKE_DIRECTORY "${spv_dir}")
    set(_entries)
    set(_spvs)
    set(_reflections)
    foreach(_spec IN LISTS spec_list)
        string(REGEX REPLACE "\\|.*$" "" _src_path "${_spec}")
        string(REGEX REPLACE "^.*\\|" "" _entry "${_spec}")
        set(_spv "${spv_dir}/${_entry}.spv")
        set(_reflect_args)
        set(_reflect_output)
        if(abi_header)
            set(_reflect_output "${spv_dir}/${_entry}.json")
            set(_reflect_args -reflection-json "${_reflect_output}")
            list(APPEND _reflections "${_reflect_output}")
        endif()
        set(_slang_entry "${_entry}")
        set(_extra_caps)
        if(_entry MATCHES "_f16kv$")
            string(REGEX REPLACE "_f16kv$" "_f32" _slang_entry "${_entry}")
            list(APPEND _extra_caps -DVELRI_F16_KV=1)
        endif()
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
            OUTPUT "${_spv}" ${_reflect_output}
            COMMAND "${VELRI_SLANGC}" "${_src_path}"
                -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/vk"
                -target spirv -profile spirv_1_6 -Wno-41012 -fvk-use-c-layout -O3
                ${_extra_caps} ${_reflect_args}
                -entry "${_slang_entry}" -stage compute -o "${_spv}"
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
    if(abi_header)
        add_custom_command(
            OUTPUT "${abi_header}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUTS=${_reflections}" "-DOUTPUT=${abi_header}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vk_abi.cmake"
            DEPENDS ${_reflections} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vk_abi.cmake"
            COMMENT "Generating Vulkan push-constant layouts from Slang reflection"
            VERBATIM
        )
    endif()
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
        "${_vk}/decode_fusion.slang|vk_qkv_rope_cache_f32"
        "${_vk}/decode_fusion.slang|vk_qkv_rope_cache_f16kv"
        "${_vk}/decode_fusion.slang|vk_rms_qkv_f32"
        "${_vk}/decode_fusion.slang|vk_rms_qkv_f16kv"
        "${_vk}/decode_fusion.slang|vk_rms_gated_gemv_f32"
        "${_vk}/decode_fusion.slang|vk_gated_gemv_f32"
        "${_vk}/decode_fusion.slang|vk_gemv_residual_f32"
        "${_vk}/gemv.slang|vk_gemv_f16"
        "${_vk}/gemv.slang|vk_gemv_bf16"
        "${_vk}/gemv.slang|vk_gemv_w8a8"
        "${_vk}/gemm_f32.slang|vk_gemm_f32"
        "${_vk}/gemm_short.slang|vk_gemm_short_f32"
        "${_vk}/gemv.slang|vk_gemv_rows_f32"
        "${_vk}/gemm_f32.slang|vk_splitk_reduce"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_f16"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_bf16"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_w8a8"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_f16_64"
        "${_vk}/gemm_coopmat.slang|vk_gemm_f32_as_f16"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_bf16_64"
        "${_vk}/gemm_coopmat.slang|vk_gemm_coopmat_w8a8_64"
        "${_vk}/gather_copy.slang|vk_embedding_f32"
        "${_vk}/gather_copy.slang|vk_copy_last_row_f32"
        "${_vk}/attention.slang|vk_kv_append_f32"
        "${_vk}/attention.slang|vk_kv_append_f16kv"
        "${_vk}/attention.slang|vk_attention_decode_f32"
        "${_vk}/attention.slang|vk_attention_decode_f16kv"
        "${_vk}/attention.slang|vk_attention_prefill_f32"
        "${_vk}/attention.slang|vk_attention_prefill_f16kv"
        "${_vk}/flash_attention.slang|vk_flash_attention_f32"
        "${_vk}/flash_attention.slang|vk_flash_attention_f16kv"
        "${_vk}/flash_attention_coop.slang|vk_prefill_coop_f16kv"
        "${_vk}/flash_attention_coop_half.slang|vk_prefill_coop_half_f16kv"
        "${_vk}/flash_attention_coop_nv.slang|vk_prefill_coop_nv_f16kv"
        "${_vk}/flash_attention_split.slang|vk_prefill_tile_f32"
        "${_vk}/flash_attention_split.slang|vk_prefill_split_f32"
        "${_vk}/flash_attention_split.slang|vk_prefill_split_f16kv"
        "${_vk}/flash_attention_split.slang|vk_prefill_merge_f32"
        "${_vk}/attention.slang|vk_attention_wide_f32"
        "${_vk}/attention.slang|vk_attention_wide_f16kv"
        "${_vk}/attention.slang|vk_attention_splitkv_f32"
        "${_vk}/attention.slang|vk_attention_splitkv_f16kv"
        "${_vk}/attention.slang|vk_attention_splitkv_reduce_f32"
        "${_vk}/attention_decode_grouped.slang|vk_attention_grouped_f16kv"
        "${_vk}/attention.slang|vk_attention_merge64_f32"
        "${_vk}/batched_linear.slang|vk_batched_linear_nt_f32"
        "${_vk}/batched_linear.slang|vk_batched_linear_nn_f32"
    )
    set(VELRI_VK_ABI_HEADER "${VELRI_SLANG_GEN_DIR}/velri_vk_abi.h")
    velri_compile_vk_shaders(VELRI_VK_SHADER_HEADER
        "${VELRI_SLANG_GEN_DIR}/spv"
        "${VELRI_SLANG_GEN_DIR}/velri_vk_shaders.h"
        "${VELRI_VK_PROD_SPECS}" "${VELRI_VK_ABI_HEADER}"
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
            "${VELRI_VK_CEIL_SPECS}" ""
        )
    endif()
endif()
