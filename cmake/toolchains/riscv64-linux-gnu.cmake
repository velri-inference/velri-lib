# Linux -> RISC-V 64 cross-compilation toolchain using Clang + target-neutral Slang LLVM IR.
# Debian/Ubuntu packages: clang, gcc-14-riscv64-linux-gnu, libc6-dev-riscv64-cross, qemu-user.
# Pass -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" to enable RISC-V Vector 1.0 (RVV) auto-vectorization.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
endif()
set(CMAKE_C_COMPILER_TARGET riscv64-linux-gnu)
set(CMAKE_C_FLAGS_INIT "-march=rv64gc -mabi=lp64d")

set(VELRI_QEMU_RISCV64 qemu-riscv64 CACHE FILEPATH "RISC-V user-mode emulator")
set(VELRI_RISCV_RUNTIME /usr/riscv64-linux-gnu CACHE PATH "RISC-V runtime for QEMU")
set(CMAKE_CROSSCOMPILING_EMULATOR "${VELRI_QEMU_RISCV64};-L;${VELRI_RISCV_RUNTIME}")
