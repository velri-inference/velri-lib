# The first C GEMV/GEMM module

[Русский](../RU/cpu-linear.md)

**Extraction status:** this page describes the retained FP32-input / ordered-FP64 prototype. Its bitwise contract remains for regression checks; it is not the numerical contract of all future formats. The [library boundary and target formats](architecture.md) define the next implementation.

Velri now has a standalone **internal CPU linear kernel** in `src/backend/cpu/` in Velri-lib.
It computes `Y = X × Wᵀ + bias`, with one input row for GEMV or several for GEMM.
This is a tested building block for the C engine. The graph, worker pool, public
application API and Vulkan backend are not implemented by this module.

## Build and check

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

The build requires CMake 3.21 or later and Clang 18+ or GCC 14+ in C23 mode.
Configuration checks that the compiler accepts `-std=c23`. There are no downloaded dependencies. The
library needs neither Rust nor LLVM at runtime. Clang uses LLVM during compilation;
GCC also builds the same C source.

`VELRI_CPU_MULTIVERSION=OFF` builds only the baseline variant. With the option
enabled on x86-64, AVX2/FMA and AVX-512F/FMA variants are built too. CPU and OS
support are checked before selecting them. The dispatcher itself uses baseline
instructions. `AUTO` chooses the widest available variant; it is not an autotuner
and does not promise the best speed on every processor.

The same algorithm builds for ARM64. Local verification includes ARM64 under
QEMU; that checks correctness, not native ARM speed. Windows and Android remain
target platforms, not verified builds of this first module. CI covers native
Linux, sanitizers, the baseline-only build and Linux ARM64 under QEMU.

## What “the same result” means

Every output follows this fixed sequence:

1. Start an FP64 accumulator at `-0.0`.
2. In increasing K order, convert the two FP32 inputs to FP64, multiply them,
   and add the product to that accumulator.
3. Add bias in FP64 after the entire dot product, or `+0.0` when bias is absent.
4. Round once to FP32.

The optimized code must match the independent reference **bit for bit**. There
is no numerical tolerance, split K reduction, or FP32 partial accumulator.
Signed zeros, subnormals and overflow are included in the tests. The caller must
use round-to-nearest/ties-to-even, gradual underflow and disabled floating-point
traps. The kernel does not change the caller's floating-point environment;
exception flags are not part of the output contract.

NaN outputs use exactly `0x7fc00000`. This makes their representation consistent
across variants. It explicitly refines the old Rust oracle's unspecified NaN
payload behavior; Rust does not promise portable or deterministic NaN result
bits. We do **not** claim to preserve those old payloads.
[Rust floating-point documentation](https://doc.rust-lang.org/std/primitive.f32.html#nan-bit-patterns).

The ordinary, row-major C reference is in `tests/cpu/reference.c`. It shares no
packing or arithmetic helpers with the fast implementation. A frozen binary
fixture from the unmodified historical Rust reference checks both implementations;
its provenance and hash are in `tests/cpu/fixtures/README.md`.

The previous FP32 fast kernels used tolerance checks and do not satisfy this
new exactness requirement. Their historical TFLOPS values cannot be presented
as performance of this C implementation.

## How the kernel gets faster

The common source is `linear_kernel.c`. CMake compiles it with different tile
sizes and target instructions, then generates the variant table. Small AVX2
arithmetic tiles live in `linear_avx2.h`; the API, packing and numerical contract
are shared. The portable C path also builds for ARM64. No arithmetic code checks
a processor-model name.

SIMD lanes calculate different output features. Each output retains its original
K order. Fully unrolling the short column loop lets the compiler hold sums in
registers instead of repeatedly writing them to the stack.

FMA can combine the FP64 multiply and addition here without changing finite
results: two FP32 values have at most 24 significant bits each, so their product
needs at most 48 bits and fits exactly in FP64's 53 bits. Its exponent also fits.
The reference keeps separate multiply and add; the optimized path may fuse them.
This argument does not allow reassociating the sum or using FP32 FMA instead.

AVX2 GEMV loads four FP32 weights directly into each FP64 vector. This avoids
GCC's extra wide loads and vector-half extraction in the original scalar C loop.
The same primitive handles an odd GEMM row. With C=32, K≤2048 and at least eight
output tiles per call, GEMV first expands the input into a bounded 16 KiB stack
buffer. Other sizes read the original input directly. The copy is included in
execution time, is private to the call, and does not change prepared weights.

The main AVX2 GEMM path processes three rows by sixteen columns. Its inner loop
uses twelve accumulator registers, three input registers and one weight register.
A small assembly primitive fixes this assignment: a compiler-generated 3×16 loop
spilled an accumulator on every K step. The matrix traversal remains C. It groups
up to 24 input rows and 128 K entries, reusing a 16 KiB weight panel. FP64 partial
sums take 6 KiB and temporary expanded inputs take 3 KiB, plus a small row-tail
buffer. Sums carry across K panels without rounding to FP32; bias is added once.

Narrower, short-K and other ISA variants retain the common row-pair path. Every
path handles incomplete tiles and naturally aligned inputs. Bias and no-bias
finishing loops are separate, avoiding unnecessary masked loads when bias is absent.
Tests include cancellation across K panels, row-group boundaries, special values,
and inaccessible guard pages; sanitizers alone cannot check assembly loads.

## Preparation and memory

`include/velri_lib/linear.h` defines three operations:

| Operation | Responsibility |
| :--- | :--- |
| `velri_linear_query` | Validate M/K, select a supported variant and tile, report packed size |
| `velri_linear_pack` | Rearrange caller-provided W once into caller-provided storage |
| `velri_linear_run` | Validate views and execute using the existing packing |

W has M rows of K FP32 values. The packed order is `[column tile][K][column
within tile]`. For tile width C, element `(c, i)` is stored at
`(c / C) * C * K + i * C + c % C`. Unused columns in the last tile contain zero.
The required capacity is `ceil(M / C) * C * K` floats.

The selected tile fits at least once in M. Thus packing uses less than twice
the actual weight count; M=1 uses exactly K floats. The plan reports all padding.
Weights stay FP32: there is no FP64 weight copy. The original W may be released
after packing, but both allocations count while they are simultaneously alive.

The caller owns the plan, packing, inputs and outputs. A plan must come from
`query` and must not be modified. A packing belongs to the plan that produced
it; equal byte counts alone do not make layouts interchangeable. Plans and
packed weights may be shared by concurrent calls with separate outputs.
Execution allocates no heap memory and does not repack weights.

M and K must be positive; zero input rows are allowed. Counts and strides are
in float elements. Strides must cover a logical row, including when rows=0.
Only natural float alignment is required. Read-only views may overlap each
other. Output must not overlap an input view, its row gaps, or the plan.
Packing requires separate source and destination regions. Size overflow,
insufficient capacities and forbidden overlap are rejected before any write.

This internal API does not create threads. A future scheduler can partition
validated complete column tiles into disjoint output ranges. The direct kernel
function requires those checks to have already succeeded; ordinary callers use
`velri_linear_run`. The common backend interface remains a separate design.

## Measure without losing the correctness check

```sh
cmake -S . -B build -DVELRI_BUILD_BENCHMARK=ON
cmake --build build
./build/bench_linear auto 1 1024 1024 1000 21
```

Arguments are variant, input rows, M, K, repetitions per sample and sample count.
The optional final argument is an independently justified **arithmetic ceiling
in TFLOPS for this single-thread run**. If it is absent, the tool reports the
ceiling as unknown rather than inventing a CPU frequency or memory limit.

The tool checks every output before timing and after every sample. It reports
preparation separately and includes view validation in call time. TFLOPS counts
the useful `2 × rows × M × K` operations, excluding padded work and conversions.
Repeated fixed weights may be cached. This tool does not measure a worker pool
or a model graph; the [measurement rules](https://github.com/velri-inference/velri/blob/main/docs/EN/performance.md) still apply.

Address/undefined-behavior checks can be enabled with `VELRI_SANITIZE=ON` and a
compiler with the corresponding runtimes installed. Tests also use guard pages
and concurrent shared-weight calls on POSIX. A passing arithmetic test alone
does not establish correct memory access or thread-safe ownership.

## Check how GEMV scales across cores

Linux has a separate diagnostic with persistent workers:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVELRI_BUILD_PARALLEL_BENCHMARK=ON
cmake --build build
lscpu -e=CPU,CORE,SOCKET,NODE
./build/bench_linear_parallel auto 1024 1024 0,1,2,3 7 normal 16
```

Replace `0,1,2,3` with available CPU IDs from your machine. First choose one
logical CPU per physical core; then try adding the SMT siblings. CPU numbering
does not reliably tell you which threads share a core. The tool checks the
process affinity and rejects unavailable or duplicate IDs.

Arguments are variant, M, K, CPU list, timed sample count, `normal|huge` page
mode and the number of verification jobs. Zero timed samples performs only
correctness checks. Each verification job changes the input, alternates bias
on/off and overwrites the output before checking it against the independent
reference. Output guards are checked too.

Workers share immutable prepared FP32 weights. They receive disjoint ranges of
complete output tiles, so each output keeps its original K accumulation order.
The caller performs one range itself. Worker count is capped at the number of
tiles; a small matrix must not start idle workers just because more CPU IDs were
requested. The selected IDs and actual worker count appear in the JSON output.

Timing includes dispatch, arithmetic and waiting for completion. Packing,
allocation, thread creation and initial view validation happen beforehand.
Workers spin between calls: this deliberately measures a busy pool, not the
power use or wake-up latency of an idle inference engine. The diagnostic is not
the engine's scheduler. It does not choose a thread count automatically.

Each sample reports useful TFLOPS, median/p99/maximum call latency, process CPU
time, involuntary context switches and minor page faults. Outputs are checked
before and after every sample. Compare repeated runs in alternating order;
report slow samples and latency tails as well as medians.

`huge` requests transparent huge pages before first touch. The tool reads their
size from Linux and reports the actual `AnonHugePages` bytes for its weight
mapping. A successful request does not guarantee allocation. Compare with
`normal` on the same machine; huge pages are not a universal speedup.

An optional final argument supplies an independently measured read throughput
in **GB/s**, for the same worker count and cache/RAM working set. Its equivalent
FP32-weight GEMV rate is `GB/s / 2000` TFLOPS: two useful operations per four-byte
weight. Output names this `read_control_TFLOPS`. It is an empirical comparison,
not a physical ceiling; the tool neither measures it nor invents a value when
the argument is omitted. Do not substitute a whole-socket memory specification
for the bandwidth available to a VM.

## AVX2 validation on Colab, 10 November 2026

The AVX2 changes above were built and measured on a CPU-only Colab VM reporting
AMD EPYC 7B12, four physical cores and eight logical CPUs. Clang 18.1.3 and GCC
14.2.0 both used C23, O3 and strict arithmetic flags. Each comparison uses the
same compiler for the old and new kernel; a speedup is not a compiler comparison.
The old source is from `bf9f45b7`. The diagnostic pool, packing and allocations
are shared by both versions. Timings include dispatch and waiting, but exclude
packing, thread creation and initial API validation.

These are medians of ten alternating AB/BA pairs, each sample lasting at least
100 ms after warmup. All outputs are checked bitwise before and after timing.
The table uses the requested huge-page mode; the 256×256 and 512×512 weight
allocations are smaller than 2 MiB and actually use ordinary pages in this harness.
Actual page allocation is recorded, rather than inferred from the request.

| Workload | Compiler | Workers | Before, TFLOPS | After, TFLOPS |
| :--- | :--- | ---: | ---: | ---: |
| GEMV, M=256 K=256 | GCC | 1 | 0.01143 | 0.03583 |
| GEMV, M=1024 K=1024 | GCC | 4 | 0.04249 | 0.08948 |
| GEMV, M=1024 K=1024 | Clang | 4 | 0.10201 | 0.09699 |
| GEMV, M=8192 K=8192 | Clang | 4 | 0.02365 | 0.02349 |
| GEMM, rows=64 M=512 K=512 | Clang | 1 | 0.02788 | 0.04228 |
| GEMM, rows=64 M=512 K=512 | Clang | 4 | 0.09634 | 0.15415 |
| GEMM, rows=64 M=512 K=512 | GCC | 4 | 0.06867 | 0.14923 |
| GEMM, rows=64 M=4096 K=4096 | Clang | 4 | 0.09848 | 0.13835 |

The large GEMV gain fixes GCC's generated code. Clang GEMV has no consistent
overall speedup; some small cases regress. GEMM improves substantially with
both compilers. These measurements do not establish leadership across CPUs.

Independent controls give the following references, all expressed in TFLOPS:

| Workers | Read 4 MiB, equivalent GEMV | Read 256 MiB, equivalent GEMV | Pure FP64 FMA |
| ---: | ---: | ---: | ---: |
| 1 | 0.03796 | 0.01150 | 0.05096 |
| 2 | 0.05271 | 0.02022 | 0.10172 |
| 4 | 0.12660 | 0.03119 | 0.20319 |
| 8 | 0.12548 | 0.03464 | 0.20385 |

The read controls use confirmed huge pages and convert GB/s to equivalent GEMV
TFLOPS as described above. They and the FMA loop are measured references, not
proven attainable GEMV ceilings. The VM's physical RAM bandwidth entitlement is
unknown. Eight SMT threads share the arithmetic units of four cores; they are
not consistently faster, and the raw data retains slow samples and latency tails.

A separate comparison with PyTorch 2.14.1/MKL 2024.2 uses the same persistent
pool and output partitions, with single-thread MKL inside each worker. On four
workers, cached GEMV 1024² gives 0.09458 TFLOPS for Velri and 0.06948 for PyTorch;
GEMV 8192² gives 0.02617 and 0.02852. GEMM 64×512² gives 0.15563 and 0.24200.
PyTorch uses FP32 arithmetic and does not satisfy Velri's ordered FP64 reference
contract. Ordinary ATen calls were also measured: their GEMV used about one CPU
even when four threads were requested, so that comparison includes different
parallelization. It is not evidence of a fourfold microkernel improvement.

Final checks passed with Clang, GCC, GCC ASan/UBSan, baseline-only Clang and
ARM64 GCC under QEMU. Each available x86 variant passed 491,680 exact output/guard
comparisons. The independent reference and frozen fixture are unchanged.
AVX-512 compiled but could not execute on this VM. The complete corrected
comparison has 39 configurations and 1,560 timing samples. Its `v2` harness fixes
a metadata-printing error in an earlier run and validates dimensions and workers.
Sources, hashes, disassembly, raw samples, rejected variants and the full report
remain local in `docs/benchmarks/avx2-finish-2026-11-10/`, excluded from Git.

## Earlier exact-kernel measurements: Ryzen 9 9950X

The following measurements predate the AVX2 changes described above.

A matched comparison used the new C arithmetic source inside the archived Rust
worker-pool harness on Ryzen 9 9950X. It measures complete prepared linear calls,
not the future C runtime. Each configuration has three alternating process pairs,
21 samples per process and every output checked bitwise. The baseline is the first
**exact FP64-accumulating C version**, not the faster historical FP32 implementation.

| Workload, 16 workers | First exact version, TFLOPS | Final version, TFLOPS | FP64 arithmetic reference, TFLOPS |
| :--- | ---: | ---: | ---: |
| Cached GEMV, M=1024 K=1024, AVX-512 | 0.460 | 0.773 | 2.425 |
| Odd GEMV, M=1025 K=1023, AVX2 | 0.173 | 0.538 | 1.287 |
| GEMM, rows=64 M=512 K=512, AVX-512 | 0.497 | 1.133 | 2.448 |
| GEMM, rows=256 M=4096 K=4096, AVX-512 | 0.424 | 1.419 | 2.258 |
| Streaming GEMV, M=8192 K=8192, AVX-512 | 0.0433 | 0.0427 | RAM limit unknown |

The arithmetic references use CPU-wide APERF/MPERF observations bracketing the
final timed loops: 32 FP64 FLOP/cycle/core for AVX-512, 16 for AVX2. Counter
snapshots are staggered and include OS activity. These are conditional FMA
references, not attainable ceilings for a loop containing conversions, loads and
synchronization. The streaming case is slightly slower and has no established
physical RAM ceiling; its unrelated arithmetic reference is about 2.86 TFLOPS.

The retained AVX2 width change improves cached GEMV but loses roughly 2–3% on
the measured GEMM controls. In the final comparison, eight-worker cached GEMV is
0.539 TFLOPS against a 1.359 TFLOPS arithmetic reference. There is no universal
claim that 16 workers or the widest ISA is optimal.

A separate direct call to the shipped C API, pinned to one core with M=64 K=1024,
measured 0.0899 TFLOPS. The supplied 0.1824 TFLOPS nominal arithmetic reference
assumes 5.7 GHz on that core; it is not a measured sustained clock. This check
includes API validation and no worker pool.

Software sampling confirms substantial worker coordination and waiting CPU time
in the archived short-call harness. Sampling percentages are not latency
fractions. Hardware event profiling was unavailable (`perf_event_paranoid=4`).
A read-only XOR control is retained separately; it is not treated as a physical
memory ceiling. Raw data, source snapshots, disassembly, profiles, executable
hashes and rejected/confounded comparisons remain local under
`docs/benchmarks/linear-bottlenecks-2026-11-10/`, excluded from Git.
