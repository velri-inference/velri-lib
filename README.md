# Velri-lib — C23 Tensor, Graph & Khronos Slang AOT Compute Library

**[Website](https://velri-inference.github.io/)** · **[English Docs](docs/EN/architecture.md)** · **[Русский](docs/RU/architecture.md)** · **[Slang AOT Kernels](docs/EN/kernels.md)** · **[Velri Engine](https://github.com/velri-inference/velri)** · **[VTQ Quantization](https://github.com/velri-inference/vtq-research)** · **[llms.txt](llms.txt)** · **License: [GPL-3.0](LICENSE)**

**Velri-lib** (`VelriLib::velri`, v0.7.0) is the standalone C23 neural-network computation library powering the [Velri](https://github.com/velri-inference/velri) inference engine. It provides zero-copy tensor views, liveness scratch memory planning, UMA-aware execution graphs, reusable transformer blocks, and modular **Khronos Slang (`slangc` AOT)** compute kernels for **CPU** (x86-64 AVX2 / AVX-512 VNNI, ARM64, RISC-V 64) and **Vulkan 1.3 / SPIR-V 1.6** behind a public **C23 API** (`<velri_lib/velri_lib.h>`), with **zero Python, PyTorch, CUDA, LLVM, or Slang runtime dependencies**.

---

## Measured Throughput

### CPU Slang AOT Roofline (`bench_slang_cpu --workers 16`, AMD Ryzen 9 9950X, 16 Physical Cores)

| Operation / Mode | `1024×4096×4096` GEMM | `256×4096×4096` GEMM | Cold-DRAM GEMV | Implementation |
| :--- | :---: | :---: | :---: | :--- |
| **FP32 (`F32`)** | **`4.019 TFLOPS`** | `3.958 TFLOPS` | `66.4–71.3 GB/s` | 5-loop L2-resident BLIS in portable Slang AOT |
| **FP16 → FP32 (`F16`)** | **`4.067 TFLOPS`** | `4.167 TFLOPS` | `66.4–71.3 GB/s` | Shared `IElement` Slang AOT |
| **BF16 → FP32 (`BF16`)** | **`4.197 TFLOPS`** | `4.167 TFLOPS` | `66.4–71.3 GB/s` | Shared `IElement` Slang AOT |
| **F32 × BF16 (`F32xBF16`)** | **`4.156 TFLOPS`** | `4.293 TFLOPS` | `66.4–71.3 GB/s` | Mixed-precision Slang AOT |
| **FP64 (`F64`)** | **`2.022 TFLOPS`** | `2.052 TFLOPS` | `66.4–71.3 GB/s` | Zero-spill AVX-512 BLIS in portable Slang AOT |
| **W8A8 INT8 (raw / packed)** | `17.152` / **`17.866 TOPS`** | `17.028` / **`17.834 TOPS`** | `66.4–71.3 GB/s` | Exact mod $2^{32}$ arithmetic (`80.3%` VNNI peak) |

### Vulkan 1.3 Compute (`decode64`, NVIDIA RTX 5070 Ti, TinyLlama-1.1B W8A8)

- End-to-end 64-token single-request decode: **`563.16 tok/s`** (`113.64 ms` / 64 tokens).
- Weighted W8A8 projections across all 155 layers/heads: **`1.280 TOPS`** (`76.69%` of streaming memory ceiling; **`1.600 TOPS` / `95.84%`** on `32000×2048` `lm_head`).

---

## What Works Today

- **`VelriLib::velri`** (compatibility alias `VelriLib::core`, umbrella header `<velri_lib/velri_lib.h>`): single consolidated static library with public C23 headers for types, backends, graphs, and reusable blocks:
  - **Core runtime:** `Storage` / `TensorView` (zero-copy subviews), central operation validator, device-agnostic graph builder, liveness scratch arena packer, UMA-aware memory report, plan-level shared `WEIGHT` bindings (`velri_plan_bind_weight_host`), device-resident `STATE` with explicit host sync (`velri_slot_sync_state_{from,to}_host`), refcounted `ctx`/`plan` lifetimes, automatic `Device A -> RAM staging -> Device B` transfers with residence deduplication, and concurrent isolated `velri_slot` execution (`0 B` hot-path allocations).
  - **Built-in CPU backend:** 23 Slang AOT kernels (`entry_cpu.slang` compiled via `slangc` into target-neutral LLVM IR → Clang or native `.o` variants `base`, `avx2`, `avx512`, including FP64 linear and 5-loop BLIS GEMM) dispatched over persistent 1-to-1 core-pinned `velri_pool`.
  - **Built-in Vulkan backend (`-DVELRI_ENABLE_VULKAN=ON`):** 29 modular `spirv-val`-verified SPIR-V 1.6 compute shaders (`src/kernels/slang/vk/`) with `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`, tiled FlashAttention, Split-KV decode, fused QKV+RoPE and gate/up+SiLU, DAG barrier elision, lazy pipeline creation, bounded chunked staging, and pre-allocated 8-slot async `VkFence` rings.
  - **Operations & data types:** `Linear` (`F32`, `F64`, `F16`, `BF16`, `F32×BF16`, `W8A8` raw/packed), `RMSNorm`, `LayerNorm`, `RoPE`, `Attention` (causal / sliding-window / GQA / FlashAttention / Split-KV), `Embedding`, `GatedMLP` (`SiLU`, `GELU`, `ReLU2`), `Elementwise`, `Softmax`, `FWHT`, `QuantRow`, and `Copy`.
  - **Verification:** `test_kernels` (unified CPU ISAs + Vulkan correctness across all 12 ops + edge cases against independent C23 references) and `test_core_contract`.
- The earlier handwritten C23/SIMD CPU backend and deprecated `VelriLib::cpu_linear` target are archived on branch `archive/c23-cpu-legacy` in [velri](https://github.com/velri-inference/velri). Public C headers are experimental and do not promise a stable binary ABI yet.

---

## Build and Test

Requires CMake 3.21+, Ninja, Clang 18+ or GCC 14+ with C23 support, and **Khronos Slang (`slangc`)** on `PATH` (or `-DVELRI_SLANGC=/path/to/slangc`):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

Standalone builds enable tests (`test_kernels` and `test_core_contract`) by default; as a subproject, tests are opt-in via `VELRI_LIB_BUILD_TESTS`. Key CMake options:

- `VELRI_CPU_MULTIVERSION` (`ON`): build x86-64 AVX2 and AVX-512 AOT variants alongside `base`.
- `VELRI_ENABLE_VULKAN` (`OFF`): enable the built-in Vulkan 1.3 / SPIR-V 1.6 compute backend (`spirv-val` checked).
- `VELRI_BUILD_BENCHMARKS` (`OFF`): build roofline tools `bench_slang_cpu` and (with Vulkan) `bench_vulkan` + ceiling calibration shaders.
- `VELRI_SANITIZE` (`OFF`): enable ASan + UBSan on C23 host code.
- `VELRI_SLANGC`: optional explicit path to `slangc` when not on `PATH` or `~/.local/bin`.
- `VELRI_SLANG_CPU_MODE` (`AUTO`): `AUTO` (target-neutral `LLVM_IR -> Clang` when Clang matches `slangc`'s LLVM version, falling back to `OBJECT` for GCC or older Clang), `LLVM_IR`, or `OBJECT`.
- `VELRI_SLANG_TARGET_TRIPLE`: optional explicit LLVM target triple used only in `OBJECT` fallback mode.

### Decode Projection Benchmarks

Build with `VELRI_BUILD_BENCHMARKS=ON` and run the applicable commands three times to compare independent runs:

```sh
build/bench_slang_cpu --decode --workers 16 --isa avx2
build/bench_slang_cpu --decode --workers 16 --isa avx512
build/bench_vulkan --decode
```

Omit `--isa` for automatic CPU dispatch or use `--isa base` for the portable path. Each CSV row measures one public graph projection. The 1-row and 64-row cases cycle through distinct `14336 × 4096` weight buffers; the smaller 64-row case reuses a `2048 × 2048` weight matrix. Multiply-accumulate counts as two operations; `ops_per_byte` counts one logical read of each input, weight, and scale and one write of each result.

### Cross-Compilation (Linux, Windows, Android, macOS, ARM64, RISC-V 64)

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

Or from an installed package (`VelriLib 0.7.0 EXACT`):

```sh
cmake --install build --prefix /path/to/velri-lib-install
```

```cmake
find_package(VelriLib 0.7.0 EXACT CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE VelriLib::velri)
```

Consumers include `<velri_lib/velri_lib.h>` and link `VelriLib::velri` (alias `VelriLib::core`). `slangc` is required only at build time; shipped binaries have zero Slang or LLVM runtime dependency, and CPU-only builds require no Vulkan SDK or network downloads.

---

## Velri Ecosystem

- **[velri-inference/velri](https://github.com/velri-inference/velri)** — Native C23/C++23 LLM inference engine built on `Velri-lib`
- **[velri-inference/vtq-research](https://github.com/velri-inference/vtq-research)** — **VTQ** (*Velri Trellis Quant*, `.vtq` v2) trellis weight quantization format (`1.5–6.0 bpw`)
- **[velri-inference.github.io](https://velri-inference.github.io/)** — Unified documentation hub, architecture summary, and [`llms-full.txt`](https://velri-inference.github.io/llms-full.txt)

---

## License & Citation

Licensed under the **[GNU General Public License v3.0](LICENSE)**. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
