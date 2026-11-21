# Strips host-specific target metadata from LLVM IR emitted by `slangc -target llvm-ir`
# so the target C compiler (Clang) lowers the IR cleanly for any OS and CPU architecture.
if(NOT DEFINED INPUT_LL OR NOT DEFINED OUTPUT_LL)
    message(FATAL_ERROR "strip_ll_target.cmake requires -DINPUT_LL=... and -DOUTPUT_LL=...")
endif()

file(READ "${INPUT_LL}" _ir)
string(REGEX REPLACE "(^|\n)target datalayout = [^\n]*\n" "\\1" _ir "${_ir}")
string(REGEX REPLACE "(^|\n)target triple = [^\n]*\n" "\\1" _ir "${_ir}")
string(REGEX REPLACE "\"target-cpu\"=\"[^\"]*\" ?" "" _ir "${_ir}")
string(REGEX REPLACE "\"target-features\"=\"[^\"]*\" ?" "" _ir "${_ir}")
file(WRITE "${OUTPUT_LL}" "${_ir}")
