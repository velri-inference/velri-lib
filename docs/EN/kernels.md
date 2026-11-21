# Kernel Architecture: Slang-First (`slangc` AOT for GPU and CPU)

[Русский](../RU/kernels.md)

Rather than maintaining separate GLSL shaders, `#ifdef` matrices, and duplicated handwritten SIMD files for every CPU ISA, `Velri-lib`'s target kernel system is built around **Khronos Slang** (`Apache-2.0 WITH LLVM-exception`) with **pure build-time AOT compilation via `slangc`**.

**Status:** active implementation in `velri-lib/src/kernels/slang/`. `slangc` is required at build time for `VelriLib::velri` (23 CPU MatOps kernels over `velri_pool`, including FP64 linear), while `-DVELRI_ENABLE_VULKAN=ON` adds 29 modular SPIR-V 1.6 shaders in `src/kernels/slang/vk/` (plus 6 benchmark-only ceiling shaders under `tools/vk_ceilings.slang` when `VELRI_BUILD_BENCHMARKS=ON`). The earlier handwritten C23/SIMD backend is preserved for historical reference on branch `archive/c23-cpu-legacy`.

---

## 1. Core Principles: Reusable Components and Zero Duplication

1. **Small composable components instead of monolithic copies:** shared math is factored into Slang interfaces and generics (`IElement`, `IActivation`, `IEpilogue`, `IKVSource`, shared reductions). One `RMSNorm`, `RoPE`, `GatedMLP`, or `Linear` implementation reuses those building blocks instead of copy-pasting.
2. **Do not multiply near-identical kernels:**
   - **No separate kernel sources for AVX2 vs. AVX-512** (or per-dtype copy-paste): one parameterized `.slang` module compiles AOT via `slangc` for all target profiles (`base`, `avx2`, `avx512`, `aarch64`, `riscv64`, `spirv`).
   - Hardware differences (such as GPU subgroup reductions vs. CPU loops, or GPU `CoopMat`) stay isolated inside thin low-level primitives rather than duplicating whole kernels.
3. **Pure AOT (`slangc` at build time) — zero compiler at runtime:** the shipped Velri binary has no runtime dependency on `libslang`, LLVM, `shaderc`, Python, or PyTorch.

```text
src/kernels/slang/
  ├── common/ (numeric, activations, epilogue, reductions)
  └── ops/    (norm, rope, elementwise, linear, ...)
        │
        ├──► slangc -target spirv      ──► .spv (+ spirv-val)       ──► Vulkan backend
        ├──► slangc -target llvm-ir/o  ──► .o (CPU ISA variants)     ──► CPU backend (velri_pool)
        └──► slangc reflection         ──► generated CMake dispatch/binding headers
```

---

## 2. Modular Composition in Slang

Velri's composition invariants are expressed directly in Slang types with zero model-family branches:

- **`IElement` (`common/numeric.slang`):** uniform access to `F32`, `F64`, `FP16`, `BF16`, and `INT8` (load/convert to accumulator and store back) so operations are not duplicated per dtype.
- **`IActivation` (`common/activations.slang`):** `SiLU`, `GELU_tanh`, `ReLU2`, `Identity` (+ optional symmetric `clamp`) — reused simultaneously by standalone activations, `GatedMLP`, and `Linear` epilogues.
- **`IEpilogue` (`common/epilogue.slang`):** bias addition, scaling (`W8A8` / `W8A32`), and optional activation — shared by `GEMV` and `GEMM` across CPU and GPU.
- **`IKVSource`:** key/value source for attention — separate buffers, one shared buffer, or on-the-fly reconstruction `K = RoPE(V * w_k)` (Gemma 4) without allocating a temporary `K` buffer.
- **AOT fusions:** hot sequences (`Residual + RMSNorm`, `GatedMLP`, `Linear + Epilogue`, `K=V + Attention`) are instantiated as AOT specializations of the same interfaces from a controlled CMake manifest.
- **Autodiff (`bwd_diff`):** built-in Slang capability available when needed for future quantization/calibration tools without rewriting math.

---

## 3. GPU Path: Vulkan / SPIR-V and Multi-GPU Portability

- **Baseline profile:** Vulkan 1.1 compute with standard storage buffers and explicit device capability queries prior to kernel selection.
- **Accelerated paths (SPIR-V 1.5/1.6):** `linalg::CoopMat` (`SPV_KHR_cooperative_matrix`, FP16/BF16/INT8), `VK_KHR_shader_bfloat16`, `VK_KHR_shader_float16_int8`, `VK_KHR_shader_integer_dot_product`, and subgroup ops are enabled **only when confirmed supported by the device**.
- **Independent capability checks:** storage, arithmetic, and accumulation capabilities are queried separately; no fixed subgroup size (32 or 64) is assumed.
- **Build-time validation:** every emitted `.spv` is validated by `spirv-val` for the target Vulkan profile in CMake and CI.
- **Pipeline management:** AOT specialization selection and `VkPipeline` creation happen in `prepare`; on-disk pipeline cache compatibility is checked against device and driver identity.
- **Portability to future GPU APIs:** the same `.slang` modules can compile to **Metal (MSL)**, **CUDA**, **DXIL**, and **WGSL** behind thin device backends in `src/backend/<name>/` without rewriting math.

---

## 4. CPU Path: Slang-First and Slang CPU Backend Improvement Plan

### Role of Slang and C23 on CPU
The target CPU strategy is **Slang-first for all optimized kernels** from the shared modular codebase. Pure **C23** retains:
- the public API, graph, memory planner, and backend registry;
- the persistent worker pool `velri_pool` and L3-aware scheduler;
- **simple independent C23 reference implementations** for correctness checks without involving the Slang compiler.

### Out-of-the-box Slang CPU limits and improvement strategy
Slang supports CPU emission via LLVM (`slang-llvm`: LLVM IR / `.o`) and C++. We account for its practical limits explicitly:
1. The C++ backend lacks workgroup barriers (`GroupMemoryBarrierWithGroupSync`) and general user atomics — `groupshared`-synchronized GPU shaders do not port directly to CPU.
2. Default C++ output carries wrapper overhead unsuitable for short `velri_pool` dispatches; CPU targets need direct **LLVM IR / `.o` lowering with a clean C-ABI** and `0 B` heap allocations.
3. One Slang source compiles to target LLVM IR, and **compiler target flags** (`-march=x86-64`, `-mavx2 -mfma -mf16c`, `-mavx512f -mavx512bw ...`, `aarch64`, `riscv64`) emit ISA variants — **without duplicating kernel code for AVX2 vs. AVX-512**.
4. Hot CPU `GEMM` / `GEMV` requires register tiling, bounded `scratch` panel packing, and hardware dot instructions (`vdpbf16ps`, `vpdpbusd`). Wherever stock Slang CPU codegen falls short, we measure the exact gap and apply targeted Slang CPU backend improvements (clean C-ABI, `fp16`/`bf16`, vector primitives/intrinsics).

### Verified Slang `2026.22.1` findings and limits
Slang CPU is always built into `VelriLib::velri` (`entry_cpu.slang`); Vulkan is opt-in via `-DVELRI_ENABLE_VULKAN=ON` (`vk/*.slang`), and both backends are verified end-to-end by `test_kernels` (with optional `bench_slang_cpu`/`bench_vulkan` under `-DVELRI_BUILD_BENCHMARKS=ON`). With `slangc 2026.22.1`, both targets are verified from shared `common/` modules:
- **CPU (`VELRI_SLANG_CPU_MODE=AUTO|LLVM_IR|OBJECT`):** zero-runtime C-ABI emission (`[DllExport] export __extern_cpp`); one parameterized `linear_cpu.slang` compiles across all OSes and CPU architectures with zero per-ISA source duplication.
  - **Primary path (`LLVM_IR`, default for compatible Clang):** `slangc -target llvm-ir -O0 -fp-mode fast` emits target-neutral LLVM IR, [strip_ll_target.cmake](../../cmake/strip_ll_target.cmake) removes host `target datalayout` / `target triple` metadata, and CMake's own `Clang` lowers `.ll` → `.o`/`.obj` with the active toolchain's target/ABI/PIC flags. This removes OS/CPU triple hardcoding and supports **Linux (x86-64, AArch64, RISC-V 64 / RVV 1.0)**, **Windows (x86-64, ARM64)**, **Android (`arm64-v8a`, `x86_64`)**, and **macOS** from one source.
  - **Automatic fallback (`OBJECT`, for GCC and older Clang):** CMake runs a quick `.ll` configure probe; if the C compiler cannot parse `slangc`'s LLVM IR version (e.g., GCC or Clang 18 against `slang-llvm` based on LLVM 21), `AUTO` seamlessly falls back to `slangc -target host-object-code -llvm-target-triple` (covering x86-64 and AArch64 in official prebuilt `libslang-llvm.so`).
  - Key LLVM codegen practices:
    - `pack_b` via `IElement::load4F32` + aligned `Ptr<float4>`/`Ptr<double4>` stores (eliminates `vgather`/`vscatter`), plus zero-copy `A` reads for native `F32`/`F64` (eliminates `LoopIdiomRecognize` `call memcpy`);
    - software IEEE-754 `ElemF16` (`-DVELRI_SOFT_F16=1`) on baseline profiles without mandatory F16C/fcvt — eliminates external `__extendhfsf2`/`__truncsfhf2` libcalls (critical for Windows MSVC and RISC-V `rv64gc`);
    - exact modulo-$2^{32}$ `W8A8` reusing the `F32` outer-product microkernel at `KC=128` ($|\sum| \le 2^{21} \ll 2^{24}$);
    - single-loop `select(v > m, v, m)` reductions (scalar `max()` and `?:` emit `vucomiss` and block vectorization; `select` yields 4-way ZMM `vmaxps`), Radix-4 Stockham ping-pong FWHT without in-place aliasing, and writing local arrays through `Ptr`.
- **Vulkan (`-target spirv -profile spirv_1_6 -fvk-use-c-layout -O3`):** under C layout Slang defaults BDA accesses to `Aligned 4` (`4× LDG.32`), so explicit `OpLoad`/`OpStore Aligned 16` helpers (`LDG.128`/`STG.128`), `ks=0 / BAR / STS.128 + ks=1` overlap on single-buffered `CoopMat` (`20 KiB` -> `2 WGs/SM`), and `Split-K` (`vk_splitk_reduce`) for small-batch `F32` are essential.
- **Remaining upstream `slang-llvm` `2026.22.1` gaps (scope for future Slang CPU backend work):**
  1. **`BFloat16` is unsupported on the LLVM target** (`InternalError: Unsupported type for LLVM target!`) — bit-shift `ElemBF16` (`uint16_t`) is used instead, preventing `vdpbf16ps` (`AVX512_BF16`) emission.
  2. **`dot4add_i8packed` and integer dot products do not lower to `vpdpbusd` (`AVX512_VNNI`) on CPU** — capping W8A8 at the FP32 FMA ceiling (`~4.3 TOPS` via `LLVM_IR -> Clang 21` vs. `~8.7 TOPS` for the archived handwritten VNNI kernel on branch `archive/c23-cpu-legacy`).
  3. **Official prebuilt `libslang-llvm.so` bundles only `X86` and `AArch64` LLVM backends:** in `OBJECT` mode (with GCC), targets such as `RISC-V 64` are unavailable; use the standard `LLVM_IR -> Clang` pipeline ([riscv64-linux-gnu.cmake](../../cmake/toolchains/riscv64-linux-gnu.cmake)), where Clang compiles `.ll` directly to `rv64gc` or `rv64gcv` (RVV 1.0).

### Acceptance rules for new Slang kernels
Every new or optimized Slang kernel is accepted only after it:
- passes numerical verification against the independent C23 reference (`tests/cpu/typed/reference.c`), ASan/UBSan, and ISA dispatch checks (`Nehalem`/`Haswell` under QEMU; `spirv-val` for GPU shaders);
- reports measured throughput against physical device ceilings under [performance.md](../../../docs/EN/performance.md).

---

## 5. Numerical Contracts and Verification

- **Validate before the first write:** dimensions, strides, overflow, alignment, and forbidden overlaps are rejected before touching outputs; `rows == 0` performs zero writes.
- **Zero heap allocations on the hot path:** `scratch` is queried during `prepare`.
- **Floating-point ops (`F64`, `FP32`, `FP16`, `BF16`):** FMA, multiple accumulators, blocked reductions, and polynomial approximations are permitted; bitwise equality across ISAs or CPU/GPU is not required — explicit tolerances are verified against C23 references.
- **Integer and exact ops (`INT8`, indexing):** exact deterministic contracts (`INT8 Linear` accumulates modulo $2^{32}$ without signed-overflow UB).
- **Memory safety and validation:** independent C23 references + ASan/UBSan + `guard pages` on CPU; `spirv-val` + Vulkan API/sync validation layers on GPU.
