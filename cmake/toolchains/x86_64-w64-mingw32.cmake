# Linux -> Windows x86-64 cross-compilation toolchain using MinGW-w64.
# Recommended: llvm-mingw (x86_64-w64-mingw32-clang, full C23) or GCC 14+ MinGW-w64
# (x86_64-w64-mingw32-gcc). Optionally runs tests via Wine when installed.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

if(NOT DEFINED CMAKE_C_COMPILER)
    find_program(_velri_mingw_clang NAMES x86_64-w64-mingw32-clang)
    if(_velri_mingw_clang)
        set(CMAKE_C_COMPILER "${_velri_mingw_clang}")
    else()
        set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
    endif()
endif()
if(NOT DEFINED CMAKE_RC_COMPILER)
    find_program(_velri_mingw_rc NAMES x86_64-w64-mingw32-windres llvm-windres)
    if(_velri_mingw_rc)
        set(CMAKE_RC_COMPILER "${_velri_mingw_rc}")
    endif()
endif()

set(VELRI_SLANG_TARGET_TRIPLE "x86_64-w64-windows-gnu" CACHE STRING "" FORCE)
if(EXISTS "/usr/x86_64-w64-mingw32/include")
    set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
endif()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Static link GCC runtime (no-op for llvm-mingw/compiler-rt) so .exe needs no extra MinGW DLLs.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libgcc")

find_program(VELRI_WINE NAMES wine64 wine)
if(VELRI_WINE)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${VELRI_WINE}")
endif()
