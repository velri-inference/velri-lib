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

All production SPIR-V modules require `shaderInt64` for device-address arithmetic. Device enumeration checks this feature and device creation enables it. The Vulkan build emits Slang reflection for each production entry point. CMake uses it to generate the C push-constant structs and compile-time checks for size and every field offset.

For F32 prompt chunks of 3–16 rows, the aligned matrix-vector kernel reuses a weight vector across up to four input rows. For 17–64 rows, aligned inputs with K divisible by 16 use a 64×64 output tile and a double-buffered 16-element K tile. Split-K can distribute the inner dimension and applies bias after reduction.

Aligned F32 prefill on 32-lane subgroups uses a tiled online softmax. With at least 128 query rows the FlashAttention kernel reuses each K/V load across four queries for head dimensions 32 and 64, or two queries for dimension 128. The corresponding key tiles contain 256 and 64 keys. Four subgroups merge their partial maxima, masses and value sums in 8320 bytes of shared memory per workgroup. Scores stay in registers and no sequence-by-sequence score buffer is allocated. Smaller chunks can use the row-wise kernel with tiles of 64 keys. Both paths honor the active causal or sliding-window range for each query. [FlashAttention, Dao et al.](https://arxiv.org/abs/2205.14135)

For F32 prefill plans with at most 64 rows, 32–64 active rows, 64-channel heads and at least 256 visible keys, a 32-lane subgroup device can use up to eight KV partitions. A workgroup shares a tile of 32 keys and values across 32 queries. A second dispatch combines the maxima, softmax masses and weighted values of the partitions. The shared device workspace holds 68 floats per query head and partition, capped at 32 MiB. Selection also requires aligned query and KV reads and 16.5 KiB of workgroup memory. The ordinary prefill kernel covers other layouts and devices and runs if optional workspace is unavailable.

Dense prefill with at least 512 query rows and 64 channels per head can reuse each KV tile across 16 neighboring queries and write normalized outputs directly. The direct F32 path uses a 32-key shared tile. Four subgroup lanes cooperate on each 64-channel dot product and reduce their softmax sums after the last tile. For F16 KV, devices supporting FP16 cooperative matrices with F32 accumulation and 32 KiB of workgroup memory can process 256 keys per workgroup iteration. Each query and softmax weight is decomposed into two FP16 parts so that the second matrix product restores the rounding residual. Softmax, accumulation and final normalization use F32. Both kernels support causal attention and windows of at least 512 positions. Their transient storage is confined to the workgroup.

With F16 KV, setting `attr.attention.allow_f16_matmul` permits rounding row-normalized queries and softmax weights to binary16 for QK and PV. Both products and the softmax accumulate in F32. The KHR cooperative kernel can process 128 or more query rows, or 64–127 rows after 256 positions of history. On devices supporting the subgroup reductions, matrix conversions and per-element operations of `VK_NV_cooperative_matrix2`, a dense causal append of at least 512 rows after 1024 positions can share each K and V fragment across two 16-row query tiles. While the subgroup evaluates softmax, it can fetch V for the current panel and K for the next. It reduces the accumulated F32 softmax weights across columns once per query row after the KV scan. This specialization requires query counts and KV capacity divisible by 32, 48 KiB of workgroup memory and aligned 64-channel heads. Preparation retains the KHR kernel for other eligible shapes.

For single-token F32 attention on 32-lane subgroups, aligned K/V layouts with equal head dimensions of 32, 64, or 128 can assign eight output channels to each workgroup. For up to 256 keys, each workgroup computes scores for its head and its slice of the value sum. If Q aliases the output, the backend selects an implementation with one writer group per head or a separate Split-KV reduction. At longer contexts Split-KV computes partial maxima, masses, and value sums and merges them with stable softmax rescaling. Prefill and other layouts retain their own selection rules.

For a long single-token decode with a binary16 cache, 64-dimensional Q/K/V heads and a group of three or four Q heads sharing one KV head, a Vulkan workgroup can read each K/V row once for the whole group. The encoder selects 64 partitions at 4096 or more visible keys if the grid would otherwise be underfilled. It checks 32-lane subgroups, 16-byte addresses and strides, shader availability and scratch capacity. An independent F32 reduction merges the 64 partials. Shorter contexts reuse the smaller recorded command buffers.

The Vulkan encoder recognizes adjacent dense F32 projections in a single-token graph. It can combine Q/K/V with full, half-layout RoPE and KV append, gate/up with SiLU and multiplication, and a projection with residual addition. A preceding weighted RMSNorm on an aligned row of up to 2048 elements can run in the projection workgroup. Group zero writes the normalized row, and the projections materialize all outputs visible in the original graph. Physical buffer-span checks exclude cross-workgroup aliases. Unsupported strides, shapes, and subgroups use the original operations.

For single-token W8A8, adjacent weighted FP32 RMSNorm and row quantization can share a workgroup. A quantization followed by a bias-free W8A8 GEMV can also share a dispatch for aligned rows up to 2048 elements. Both forms publish the intermediate results for other graph consumers. At 64 active rows, eligible cooperative-matrix projections use 64 × 32 output tiles and read 128 input bytes per channel per depth step.

---

## 4. CPU Path: Slang-First and Slang CPU Backend Improvement Plan

### Role of Slang and C23 on CPU
The target CPU strategy is **Slang-first for all optimized kernels** from the shared modular codebase. Pure **C23** retains:
- the public API, graph, memory planner, and backend registry;
- the persistent worker pool `velri_pool` and L3-aware scheduler;
- **simple independent C23 reference implementations** for correctness checks without involving the Slang compiler.

The temporary CPU W8A8 GEMM path is enabled by default. Set `-DVELRI_C23_W8A8=OFF` to build the Slang-only path. For batched matrices (`rows >= 4`) with channel count divisible by 64 and depth divisible by 64 and at least 256, C23 computes tiled GEMM over signed 32-channel panels (`ROW_BLOCK = 128`) on every supported CPU. The x86 dispatcher can also select AVX2 or AVX-512 VNNI. VNNI uses biased (+128) weights interleaved in depth quartets across 64 channels, a 6 × 64 (plus 4 × 64 / 2 × 64 / 1 × 64) 4× quartet (16-byte) K-unrolled microkernel, `ROW_BLOCK = 256`, and reserves exactly M × K bytes for prepared weights before allocator alignment. It uses the same packed bytes for decode and narrow slices (`rows < 4`). Other shapes and all raw narrow-slice/decode (`rows < 4`) projections run directly in portable Slang.

All dense single-row float projections (`F32`, `F64`, `F16->F32`, `BF16->F32`, `F32xBF16`) on vectorized x86 targets stream one weight row at a time from depth 256, and narrow-slice raw W8A8 projections (`rows < 4`) in Slang (`w8a8_dot.slang` and `linear_cpu.slang`) stream one channel at a time for large matrices (`K >= 1024`, slice > 128 KiB) when input and output buffers do not overlap. The dispatcher checks physical output overlap for all W8A8 projections and streaming float GEMV before changing traversal order, and `velri_pool` coordinates 1-to-1 physical-core pinning across pools on symmetric Linux topologies (skipping heterogeneous CPUs and oversubscribed pools) while excluding worker cores from the caller thread's affinity mask during parallel sections.

### Out-of-the-box Slang CPU limits and improvement strategy
Slang supports CPU emission via LLVM (`slang-llvm`: LLVM IR / `.o`) and C++. We account for its practical limits explicitly:
1. The C++ backend lacks workgroup barriers (`GroupMemoryBarrierWithGroupSync`) and general user atomics — `groupshared`-synchronized GPU shaders do not port directly to CPU.
2. Default C++ output carries wrapper overhead unsuitable for short `velri_pool` dispatches; CPU targets need direct **LLVM IR / `.o` lowering with a clean C-ABI** and `0 B` heap allocations.
3. One Slang source compiles to target LLVM IR, and **compiler target flags** (`-march=x86-64`, `-mavx2 -mfma -mf16c`, `-mavx512f -mavx512bw ...`, `aarch64`, `riscv64`) emit ISA variants — **without duplicating kernel code for AVX2 vs. AVX-512**.
4. Write CPU compute kernels in portable Slang. Express tiling, packing and arithmetic in the shared source. The compiler selects machine instructions for each target. Outside the temporary C23 W8A8 path, CPU kernels exclude handwritten assembly, embedded LLVM IR and ISA-specific intrinsics.

### Verified Slang `2026.22.1` findings and limits
Slang CPU is always built into `VelriLib::velri` (`entry_cpu.slang`); Vulkan is opt-in via `-DVELRI_ENABLE_VULKAN=ON` (`vk/*.slang`), and both backends are verified end-to-end by `test_kernels` (with optional `bench_slang_cpu`/`bench_vulkan` under `-DVELRI_BUILD_BENCHMARKS=ON`). With `slangc 2026.22.1`, both targets are verified from shared `common/` modules:
- **CPU (`VELRI_SLANG_CPU_MODE=AUTO|LLVM_IR|OBJECT`):** zero-runtime C-ABI emission (`[DllExport] export __extern_cpp`); one parameterized `linear_cpu.slang` compiles across all OSes and CPU architectures with zero per-ISA source duplication.
  - **Primary path (`LLVM_IR`, default for compatible Clang):** `slangc -target llvm-ir -O0 -fp-mode fast` emits target-neutral LLVM IR, [strip_ll_target.cmake](../../cmake/strip_ll_target.cmake) removes host `target datalayout` / `target triple` metadata, and CMake's own `Clang` lowers `.ll` → `.o`/`.obj` with the active toolchain's target/ABI/PIC flags. This removes OS/CPU triple hardcoding and supports **Linux (x86-64, AArch64, RISC-V 64 / RVV 1.0)**, **Windows (x86-64, ARM64)**, **Android (`arm64-v8a`, `x86_64`)**, and **macOS** from one source.
  - **Automatic fallback (`OBJECT`, for GCC and older Clang):** CMake runs a quick `.ll` configure probe; if the C compiler cannot parse `slangc`'s LLVM IR version (e.g., GCC or Clang 18 against `slang-llvm` based on LLVM 21), `AUTO` seamlessly falls back to `slangc -target host-object-code -llvm-target-triple` (covering x86-64 and AArch64 in official prebuilt `libslang-llvm.so`).
  - Key LLVM codegen practices:
    - `pack_b` via `IElement::load4F32Aligned` + 8×8 `Ptr<float4>`/`Ptr<double4>` transposes (eliminates `vgather`/`vscatter` on aligned weights), zero-copy `A` reads for native `F32`/`F64` (eliminates `LoopIdiomRecognize` `call memcpy`), precomputed row pointers with 4× K-unrolled `6×64` F32 microkernels, `4×NR` tails, direct-to-`Y` final-K stores, and a 5-loop L2-resident BLIS path (`rows > 264`) that packs multi-tile `B` panels once per `MB` block and converts non-native `F16`/`BF16` `A` panels once across `NC` column strips;
    - software IEEE-754 `ElemF16` (`-DVELRI_SOFT_F16=1`) on baseline profiles without mandatory F16C/fcvt — eliminates external `__extendhfsf2`/`__truncsfhf2` libcalls (critical for Windows MSVC and RISC-V `rv64gc`);
    - the default LLVM W8A8 GEMM accumulates exact sums modulo $2^{32}$ using the `F32` outer-product microkernel at `KC=128` ($|\sum| \le 2^{21} \ll 2^{24}$);
    - single-loop `select(v > m, v, m)` reductions (scalar `max()` and `?:` emit `vucomiss` and block vectorization; `select` yields 4-way ZMM `vmaxps`), Radix-4 Stockham ping-pong FWHT without in-place aliasing, and writing local arrays through `Ptr`.
- **Vulkan (`-target spirv -profile spirv_1_6 -fvk-use-c-layout -O3`):** under C layout Slang defaults BDA accesses to `Aligned 4` (`4× LDG.32`), so explicit `OpLoad`/`OpStore Aligned 16` helpers (`LDG.128`/`STG.128`), `ks=0 / BAR / STS.128 + ks=1` overlap on single-buffered `CoopMat` (`20 KiB` -> `2 WGs/SM`), and `Split-K` (`vk_splitk_reduce`) for small-batch `F32` are essential.
- **Remaining upstream `slang-llvm` `2026.22.1` gaps (scope for future Slang CPU backend work):**
  1. **`BFloat16` is unsupported on the LLVM target** (`InternalError: Unsupported type for LLVM target!`) — bit-shift `ElemBF16` (`uint16_t`) is used instead, preventing `vdpbf16ps` (`AVX512_BF16`) emission.
  2. With this compiler version, portable integer dot expressions on CPU can lower to separate multiplies and adds. The generic W8A8 GEMV uses signed byte products and unsigned 32-bit accumulation. Its packed implementation scans all 32 channels of a panel together. The generic GEMM reuses the shared floating-point microkernel for bounded, exactly representable K-block sums and combines those sums modulo $2^{32}$.

  3. **Official prebuilt `libslang-llvm.so` bundles only `X86` and `AArch64` LLVM backends:** in `OBJECT` mode (with GCC), targets such as `RISC-V 64` are unavailable; use the standard `LLVM_IR -> Clang` pipeline ([riscv64-linux-gnu.cmake](../../cmake/toolchains/riscv64-linux-gnu.cmake)), where Clang compiles `.ll` directly to `rv64gc` or `rv64gcv` (RVV 1.0).

On native Linux x86-64 multiversion builds, `VELRI_SLANG_CPU_DOT` probes compilers for `ops/w8a8_dot.slang`. It first tries Slang LLVM IR with Clang, requesting vectorization over 64 consecutive input bytes. If that compiler cannot produce a 512-bit byte dot, it tries the Slang C++ target with G++. Each probe verifies the generated instruction and links the object with the project C compiler. Set `VELRI_SLANG_DOT_CLANG` to select a separate native Clang for this kernel, or set it to `OFF` to try G++ directly. The CPU device selects the kernel for the AVX-512 profile on hardware with AVX-512 VNNI. The batched kernel computes 3 × 8 outputs per full tile for both prepared and raw weights (converting each signed weight byte to an unsigned byte by adding 128 and subtracting `128 * sum(x)` per row modulo $2^{32}$, with raw weights XOR-packed one L1-resident tile at a time up to `KC = 4096` so both paths share the same microkernel). Narrowing each exact byte product (`[-32640, 32385]`) through `int16_t` and separating the 64-byte-aligned main loop from the scalar remainder ensures GCC emits 24/24 `vpdpbusd` instructions with zero `vpdpwssd` fallback and zero epilogue spills. With the temporary C23 path disabled, `-DVELRI_SLANG_CPU_DOT=OFF` selects the standard Slang AOT pipeline for W8A8. Arithmetic in this Slang profile is written in portable Slang.

The byte-dot GEMM collects integer sums for up to 128 rows × 32 channels (16 KiB) in a private panel per active worker, then applies the row correction and output scales in a vectorized pass over consecutive output channels. Slices smaller than three rows or four channels, zero depth, and overlapping input/output bindings use the direct single-row tile traversal (`1 × 8`, `1 × 4`, `1 × 1`).

Pass `VELRI_PLAN_PACK_CPU_WEIGHTS` to `velri_plan_prepare_ex` to pack eligible CPU W8A8 matrices at bind time. For shapes handled by Slang, the generic CPU kernels store signed bytes in panels of 32 output channels. The byte-dot profile stores each weight row as unsigned bytes with a bias of 128. It rounds the row stride up to 64 bytes when the padding is at most 1/32 of K and the rounded stride fits in 32 bits, and adds a further 64 bytes when the rounded stride is a multiple of 512 bytes (while still staying within 1/32 of K) to avoid power-of-two L1D set aliasing. The padding contains zeros. With contiguous activations (`xs == k`, `k >= 64`), a tile can include the 64-byte-rounded depth in its reduction when the next activation row belongs to the same slice. The last tile uses the logical K. For an M × K matrix the generic plan reserves `round_up(M, 32) * K` bytes. The byte-dot plan reserves the larger of this amount and `M * packed_stride`, plus allocator alignment. Check `velri_plan_weight_is_packed` before releasing the source after a successful bind. Slots and compatible plans share the packed storage. Weights used by a transfer or through incompatible views retain the original layout.

### Acceptance rules for new Slang kernels
Every new or optimized Slang kernel is accepted only after it:
- passes numerical verification against the independent C23 reference (`tests/cpu/typed/reference.c`), ASan/UBSan, and ISA dispatch checks (`Nehalem`/`Haswell` under QEMU; `spirv-val` for GPU shaders);
- reports measured throughput against physical device ceilings under [performance.md](https://github.com/velri-inference/velri/blob/main/docs/EN/performance.md).

---

## 5. Numerical Contracts and Verification

- **Validate before the first write:** dimensions, strides, overflow, alignment, and forbidden overlaps are rejected before touching outputs; `rows == 0` performs zero writes.
- **Zero heap allocations on the hot path:** `scratch` is queried during `prepare`.
- **Floating-point ops (`F64`, `FP32`, `FP16`, `BF16`):** FMA, multiple accumulators, blocked reductions, and polynomial approximations are permitted; bitwise equality across ISAs or CPU/GPU is not required — explicit tolerances are verified against C23 references.
- **Integer and exact ops (`INT8`, indexing):** exact deterministic contracts (`INT8 Linear` accumulates modulo $2^{32}$ without signed-overflow UB).
- **Memory safety and validation:** independent C23 references + ASan/UBSan + `guard pages` on CPU; `spirv-val` + Vulkan API/sync validation layers on GPU.

A linear operation with F32 tensors can opt into binary16 operands with `attr.linear.allow_f16_matmul`. On a supported Vulkan device, projections with at least 64 active rows use a 64 × 64 cooperative matrix tile. The kernel converts values while loading the tile, accumulates in F32 and writes F32 output. This requires no persistent half-precision weight copy. Omitting the flag preserves full F32 arithmetic.
