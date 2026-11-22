# Velri-lib

`velri-lib/` is the computation library inside the [Velri](../README.md) repository. It owns tensor views, storage and memory planning, kernel launch graphs, composable **Khronos Slang (`slangc` AOT)** kernels, and device backends (**CPU** and **Vulkan** first) behind a public **C23 API** (`include/velri_lib/`). Internals may use C23 and C++23, with no Python or PyTorch dependency.

Documentation: **[English](docs/EN/architecture.md)** · **[Русский](docs/RU/architecture.md)** · **[Kernels (Slang AOT)](docs/EN/kernels.md)**

---

## What Works Today

- **`VelriLib::velri`** (compatibility alias `VelriLib::core`, umbrella header `<velri_lib/velri_lib.h>`): single consolidated static library with public C23 headers for types, backends, graphs and reusable blocks:
  - Core runtime: `Storage` / `TensorView` (zero-copy subviews), central operation validator, device-agnostic graph builder, liveness scratch arena packer, UMA-aware memory report, plan-level shared `WEIGHT` bindings (`velri_plan_bind_weight_host`), device-resident `STATE` with explicit host sync (`velri_slot_sync_state_{from,to}_host`), refcounted `ctx`/`plan` lifetimes, automatic `Device A -> RAM staging -> Device B` transfers with residence deduplication, and concurrent isolated `velri_slot` execution (`0 B` hot-path allocations).
  - Built-in CPU backend: 23 Slang AOT kernels (`entry_cpu.slang` compiled via `slangc` into target-neutral LLVM IR → Clang or native `.o` variants `base`, `avx2`, `avx512`, including FP64 linear) dispatched over persistent `velri_pool`.
  - Built-in Vulkan backend (`-DVELRI_ENABLE_VULKAN=ON`): Vulkan 1.3 / SPIR-V 1.6 compute backend with `spirv-val`-verified modular Slang shaders (`src/kernels/slang/vk/`), lazy pipeline creation, bounded chunked staging, and pre-allocated 8-slot async `VkFence` ring supporting multi-op `submit_batch`.
  - Verification: `test_kernels` (unified CPU ISAs + Vulkan correctness across all 12 ops + edge cases) and `test_core_contract`.
- The earlier handwritten C23/SIMD CPU backend and deprecated `VelriLib::cpu_linear` target are archived on branch `archive/c23-cpu-legacy`. Public C headers are experimental and do not promise a stable binary ABI yet.

---

## Build and Test

Requires CMake 3.21+, Ninja, Clang 18+ or GCC 14+ with C23 support, and **Khronos Slang (`slangc`)** on `PATH` (or `-DVELRI_SLANGC=/path/to/slangc`):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

Standalone builds enable tests (`test_kernels` and `test_core_contract`) by default; as a subproject, tests are opt-in via `VELRI_LIB_BUILD_TESTS`. CMake options:
- `VELRI_CPU_MULTIVERSION` (`ON`): build x86-64 AVX2 and AVX-512 AOT variants alongside `base`.
- `VELRI_ENABLE_VULKAN` (`OFF`): enable the built-in Vulkan 1.3 / SPIR-V 1.6 compute backend (`spirv-val` checked).
- `VELRI_BUILD_BENCHMARKS` (`OFF`): build optional roofline tools `bench_slang_cpu` and (with Vulkan) `bench_vulkan` + ceiling calibration shaders.
- `VELRI_SANITIZE` (`OFF`): enable ASan + UBSan on C23 host code.
- `VELRI_SLANGC`: optional explicit path to `slangc` when not on `PATH` or `~/.local/bin`.
- `VELRI_SLANG_CPU_MODE` (`AUTO`): `AUTO` (uses target-neutral `LLVM_IR -> Clang` when Clang matches `slangc`'s LLVM version, falling back to `OBJECT` for GCC or older Clang), `LLVM_IR`, or `OBJECT`.
- `VELRI_SLANG_TARGET_TRIPLE`: optional explicit LLVM target triple used only in `OBJECT` fallback mode.

### Cross-Compilation (Any OS & CPU Architecture)

With a compatible **Clang** (`LLVM_IR` mode, automatic), `slangc` emits target-neutral LLVM IR (`.ll`) and CMake's own Clang compiles it for whatever OS and CPU architecture the toolchain targets — **Linux (x86-64, AArch64, RISC-V 64 / RVV 1.0)**, **Windows (x86-64, ARM64)**, **Android (`arm64-v8a`, `x86_64`)**, and **macOS** — without hardcoding triples. `src/platform/` selects native OS threads (`pthreads` on POSIX/Android, Win32 threads on Windows) and Vulkan dynamic loading (`libvulkan.so.1`/`libvulkan.so`, `vulkan-1.dll`, or `libvulkan.1.dylib`/`libMoltenVK.dylib`). For GCC or older Clang, `AUTO` falls back to `slangc -target host-object-code` (x86-64 and AArch64):

```sh
# Android ARM64 (or -DANDROID_ABI=x86_64), CPU + Vulkan
cmake -S . -B build-android -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DVELRI_ENABLE_VULKAN=ON
cmake --build build-android

# Linux -> Windows x86-64 (llvm-mingw / MinGW-w64 GCC 14+)
cmake -S . -B build-win64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-w64-mingw32.cmake
cmake --build build-win64

# Linux -> Linux ARM64 (aarch64-linux-gnu-gcc-14 or Clang + optional qemu-aarch64)
cmake -S . -B build-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake
cmake --build build-arm64 && ctest --test-dir build-arm64 --output-on-failure

# Linux -> RISC-V 64 (Clang LLVM_IR mode; add -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" for RVV 1.0)
cmake -S . -B build-riscv64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake
cmake --build build-riscv64 && ctest --test-dir build-riscv64 --output-on-failure
```

---

## Use from Another CMake Project

From source:

```cmake
add_subdirectory(path/to/velri-lib velri-lib-build)
target_link_libraries(my_application PRIVATE VelriLib::velri)
```

Or from an installed package (`VelriLib 0.2.0 EXACT`):

```sh
cmake --install build --prefix /path/to/velri-lib-install
```

```cmake
find_package(VelriLib 0.2.0 EXACT CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE VelriLib::velri)
```

Consumers include `<velri_lib/velri_lib.h>` and link `VelriLib::velri` (alias `VelriLib::core`). `slangc` is required only at build time; shipped binaries have zero Slang or LLVM runtime dependency, and CPU-only builds require no Vulkan SDK or network downloads.
