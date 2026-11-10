# How the GEMM and GEMV CPU kernels work

[Русский](../RU/kernel-reference.md)

**Extraction status:** this page describes the retained FP32-input / ordered-FP64 prototype. Its bitwise contract remains for regression checks; it is not the numerical contract of all future formats. The [library boundary and target formats](architecture.md) define the next implementation.

**Current C implementation:** see [CPU linear](cpu-linear.md). The historical FP32 fast paths below use tolerance checks and do not meet the new bitwise-equality contract. Their speed is not the speed of the new exact C kernels.

> Historical measurement reports, raw experimental data and old source snapshots are kept locally under `docs/benchmarks/` and excluded from Git. Plain paths to these artifacts are references, not download links.

Implementation snapshot as of **November 8, 2026**, before removal of the current code. This document preserves the formulas, data layouts, and execution rules needed to reconstruct the kernels in the new engine. The old Rust files are not needed to understand the algorithms; their names are listed at the end for comparison with the archive.

This document describes the **existing FP32 GEMM/GEMV implementations**: handwritten SIMD implementations and the common C/LLVM generator, including prepared weights. Attention, GDN, MoE, quantized multiplication, and Vulkan have different contracts and are not considered implemented by this set of kernels.

Terms are collected in the [glossary](https://github.com/velri-inference/velri/blob/main/docs/EN/glossary.md). On a first reading, it helps to follow the main problem: which numbers are multiplied, why weights are rearranged in memory, and how work is divided between threads. The address formulas and constraints alongside this explanation are needed to reconstruct the implementation precisely. General requirements for future CPU and Vulkan kernels are described in [kernels.md](https://github.com/velri-inference/velri/blob/main/docs/EN/kernels.md).

## 1. What to carry over to the new engine

For the new CPU backend, it is worth retaining **the common C/LLVM description of the computation as the primary way to implement GEMM/GEMV**, and checking its results against a simple independent reference. Handwritten AVX kernels are useful as a source of techniques and as replacements for individual parts where measurements confirm an advantage. Maintaining two complete sets of optimized matrix operations would mean fixing, testing, and porting each set to new CPUs separately.

This is a migration recommendation. The following sections describe the existing implementation; the new C engine has not yet been declared ready.

Two independent decisions must be distinguished:

| Decision | Options |
|---|---|
| How machine code is produced | Handwritten code using intrinsics, or a common description compiled by LLVM |
| How weights are stored | Ordinary rows, or output-feature stripes prepared once |

The common generator supports both weight representations. Constant weights of a linear layer benefit from preparation once and reuse across many calls. Temporary matrices and infrequent calls need the ordinary path: rearranging the entire matrix before every GEMV can consume the gain from faster computation.

LLVM is used **at build time**: the shipped engine contains no JIT, compiler, or runtime dependency on `libLLVM` for these kernels. The same generator produces separate object files for different target platforms. The resulting machine code differs between x86 and ARM.

In the old project, the common path is enabled by the `llvm-kernels` build option. The standalone `linear` function uses the handwritten implementation; `llvm_linear` and an explicitly selected LLVM executor use the generator. Comparisons therefore need to specify both the CPU instruction set and the selected implementation.

## 2. Mathematics and operation boundaries

A linear layer receives rows of input numbers X. For each output feature, it multiplies the input row by the corresponding row of weights W, sums the products, and adds an offset if bias is present. In the notation below, `r` selects the input row, `c` selects the output feature, and `i` iterates over the elements within a row:

```text
Y[r, c] = sum(i = 0 .. K-1, X[r, i] * W[c, i]) + bias[c]

X:    [N, K] — input rows
W:    [M, K] — one row of weights per output feature
Y:    [N, M] — result
bias: [M]    — optional vector
```

With `N=1`, one input row is processed: this is GEMV, a typical task when generating one token. With `N>1`, several rows are processed: this is GEMM, used for prefill and small batches, among other cases. The notation `N×M×K` denotes the three parameters of the operation, not the shape of a single three-dimensional tensor. Useful work is `2*N*M*K` FLOP, excluding bias and additional operations.

In matrix notation, this is `X * Wᵀ`, `alpha=1`, `beta=0`. Calling code cannot arbitrarily switch transposition or request addition of the previous result through `beta*Y`: the current operation does not provide those capabilities. The output is overwritten completely; its previous contents, including NaN, must not affect the result.

The following are checked before the first write:

- `M>0`, `K>0`; `N=0` is allowed and means there are no output elements.
- Input lengths are exactly `N*K`, `M*K`, `N*M`, and, if bias is present, `M`.
- Dimension multiplications, layout rounding, and byte offsets do not overflow and fit within the addressable object.
- The requested instruction variant is supported.

If any of these checks fails, the output remains unchanged. Errors that occur later during graph execution do not carry a promise to roll back all writes.

Read-only inputs may alias one another: for example, X and W may refer to the same region. The output does not overlap X, W, or bias. A future C implementation therefore cannot automatically declare all input pointers mutually independent through `restrict`.

An ordinary public call receives dense data stored row by row in row-major order. The internal kernel also accepts an output row stride `ld_y`: it writes rectangles of width `M_local` with `ld_y >= M_local`, leaving the gaps untouched. This allows different workers to fill different columns of a shared Y. The current Linear operation does not support arbitrary strides between input elements or rows.

Logical form of the internal call:

```c
kernel(x, weights, bias_or_null, y, N_local, M_local, K, ld_y);
```

This describes the existing convention; it is not an approved public ABI for the future engine. Dimensions must have the pointer width of the target platform.

## 3. Numbers, the reference implementation, and accuracy

Fast kernels store data and accumulate sums in FP32. Addition order affects rounding, so two correct kernels can produce slightly different numbers. The result depends on how partial sums are combined, the number of independent accumulators, and the partitioning of K. FMA can combine multiplication and addition with a single rounding.

Validation requires an independent reference: three nested loops directly follow the formula in the previous section. Each original FP32 value is converted to FP64, products and bias are summed in FP64, and the result is rounded to FP32 once. The reference does not use packing, SIMD tiles, or the fast-path scheduler, so an error in those mechanisms is not repeated in validation.

LLVM dot products permit reassociation of additions (`reassoc`), disregard for the sign of zero (`nsz`), and operation contraction (`contract`). Outer-product loops permit contraction of multiplication and addition. **NaN and Inf are not declared impossible**: the compiler is not allowed to assume that all inputs are finite. However, bitwise agreement is not promised for every combination of such values and every addition order.

The graph checks that results are finite separately. A NaN in one output feature must not spread to neighboring independent features because of an error in combining SIMD lanes.

The preserved comparison harness used this tolerance:

```text
abs(actual - reference) <= 5e-4 + 1e-4 * abs(reference)
```

This tolerance was used in specific FP32 tests; it does not define a proven error bound for arbitrary K and input data. Large values, subtraction of nearly equal sums, and other number formats require separate numerical checks. Tests with small binary fractions also use exact comparisons: these detect missing portions of K, bias from the wrong output, and mixed columns effectively.

## 4. What the common generator does

The generator preserves one algorithm across processors. C code handles matrix traversal, while small vector operations are described directly for LLVM:

1. The **C driver** selects GEMV, direct dot products, or blocked GEMM, traverses tiles, and handles edges.
2. **LLVM IR vector primitives** describe loads, broadcasting a value across vector lanes, multiplication/addition, reduction, and transposition. LLVM selects the final instructions and registers.

At build time, a tool asks LLVM for information about the **target** machine, rather than the build machine: the width of a fixed-width vector register and the number of registers in the corresponding class. Tile dimensions are calculated from this information. CPU model names do not appear in the formulas.

The current build sequence:

```text
target triple + features
    -> LLVM information about vectors and registers
    -> tile parameters + C driver + LLVM vector primitives
    -> Clang: C driver to LLVM IR at -O1
    -> add primitive definitions to the IR
    -> Clang: combined IR to an object file at -O3
    -> static kernel library
```

Primitive declaration names are removed before the definitions are added. In the final stage, the compiler optimizes the driver and arithmetic together. If separate calls whose bodies are invisible to the optimizer remain between them, those calls enter the inner loop and can impede acceleration.

For x86-64, a baseline profile and AVX2+FMA and AVX-512F+FMA profiles are generated. Other architectures use the baseline of their target triple. At runtime, the engine selects only a compatible profile. The generated x86 AVX profiles additionally check for AVX2/FMA support: Clang may use narrower instructions as well. The name `Scalar` when selecting a common LLVM profile means baseline, which may itself be vectorized by the compiler. The independent reference exists separately from this selection.

The tool that obtains target CPU information is currently written in C++ and uses the LLVM API **only on the build machine**. This must be considered if the entire toolchain is to be moved to C: porting the engine itself does not replace this tool. The existing build requires matching Clang and LLVM major versions, at least 18.

### How block sizes are selected

In one step, the kernel computes a small rectangle of the result. It must be large enough to reuse loaded numbers, yet fit in the available registers. The generator chooses its dimensions using the following rules.

Notation: `L` is the number of FP32 elements in a vector, `R` is the number of vector registers, `V` is the number of output-accumulator vectors per row, and `C=L*V` is the width of the main stripe.

Current rules:

```text
L = max(vector_bits / 32, 1), L is a power of two
capacity = max(R - 6, 0) / 5
V = largest power of two <= capacity; when capacity=0, use 1
C = L * V

register_rows = 6, if floor(max(R - V - 2, 0) / V) >= 6; otherwise 4
dot_columns = max(V, 4), if R >= 16; otherwise V
depth = min(128, floor(8192 / C))
block_rows = min(256, floor(16384 / C))
packing_rows = max(16, 4*log2(L))
```

The resulting dimensions are checked, including width C: temporary storage has a finite size. These rules are heuristics of the current generator. They are preserved here for reproduction, but are neither mandatory constants for the future implementation nor an exact model of CPU hardware.

In the last measured build, LLVM returned:

| Profile | L | R | V | C | Rows in the main register tile | K-block depth | Rows in the scratch block |
|---|---:|---:|---:|---:|---:|---:|---:|
| x86 baseline | 4 | 16 | 2 | 8 | 6 | 128 | 256 |
| AVX2 | 8 | 16 | 2 | 16 | 6 | 128 | 256 |
| AVX-512 | 16 | 32 | 4 | 64 | 6 | 128 | 256 |

All three profiles have `packing_rows=16`. For narrower M, a smaller tile width is selected from power-of-two vector counts. Its register tile row count increases, up to a maximum of eight: `min(register_rows * V / V_local, 8)`.

## 5. Ordinary weights: GEMV and small batches

First, consider weights in their original order. Adjacent elements of one row of W lie along K. The kernel loads several of these weights and the corresponding elements of X at once, so SIMD operates **along K**. The resulting products must eventually be summed together to obtain a single output number.

GEMV computes several outputs simultaneously, reusing a single load of X across several rows of W. It maintains several independent partial sums for each output. The processor can update them in parallel without waiting for one long chain of additions to complete.

For C_local outputs and N_local rows, the number of independent chains in the common generator is:

```text
if N_local == 1 and C_local > V:
    chains = max(4*V / C_local, 1)
else:
    chains = 4 / N_local      # N_local is 1, 2, or 4

main K-loop step = L * chains
```

For a full four-output GEMV, this gives two chains per output in the current AVX2 profile and four in AVX-512. The main output-group width is `dot_columns` for one input row and `V` for groups of 2/4 rows. Here, the count is of separate dot products; the packed-stripe width `C=L*V` has a different meaning. Output groups are processed from the main width down to smaller powers of two. Batches use groups of 4 rows, then 2 and 1: a loaded weight serves several rows of X, and a loaded X serves several outputs.

After the main loop, the partial sums for one output are combined, and then the SIMD vector lanes are summed. The K remainder is computed one element at a time, and bias is added once at the end. No load rounds K up beyond the allocated row.

In the original generator variant, the scalar remainder after the unrolled loop may include several complete SIMD vectors. The experimental `experimental-dot-kernels` first reads these remaining full vectors, then performs horizontal reduction and a short scalar tail. This variant is **not enabled by default**: improving the tail did not produce a consistent speedup across all measured shapes. It should not automatically be ported as a “corrected” version.

## 6. Ordinary weights: GEMM with panel packing

When there are many input rows, loaded weights can be reused for several rows of X. With `N >= packing_rows` and `M >= L`, the driver switches to blocked GEMM. A small panel of weights is transposed: adjacent SIMD lanes now correspond to **different outputs**, rather than adjacent K values.

Local panel:

```text
panel[i, j] = W[c+j, kb+i]
panel shape = [depth_current, C_local]
```

An inner-loop step now loads a vector of weights across output features, broadcasts a single number `X[r,i]` to all SIMD lanes, and updates a vector of outputs. No horizontal reduction across SIMD lanes is needed: each lane is already a separate output. A loaded weight vector is reused across several rows of X.

Simplified loop order:

```text
for a row block rb of at most block_rows rows:
    for a full output stripe c of width C_local:
        for a K block starting at kb:
            pack the current W panel into [depth_current, C_local]
            if kb == 0: start sums from zero
            else: continue the sums from the previous K block
            run outer-product microkernels over groups of rows
            add bias only on the last K block
        copy the completed sums to Y with the original ld_y
compute the remaining output columns using direct dot products
```

The first K block must **write** the active part of the partial-sum array `sums` before any read from it. The initial contents of temporary storage and Y do not contribute to the result. Weights may be packed again for the next block of input rows: the ordinary path temporarily rearranges only the required panel and does not create a permanent prepared matrix.

Large matrices are not copied into separately allocated heap objects on every call. Bounded temporary arrays reside on the stack: up to **32 KiB for the weight panel + 64 KiB for partial sums per active call**, up to 96 KiB in total. The main AVX2 geometry actually needs 8+16 KiB, and x86 baseline needs 4+8 KiB. For multithreaded execution, this budget must be multiplied by the number of simultaneously running kernels.

Full `L×L` squares are transposed using a sequence of SIMD-lane permutations: a shuffle network. Each step exchanges the corresponding bit of the row index and the lane index. LLVM translates this common scheme into the available instructions. The remainder of the panel depth is copied one element at a time.

## 7. Constant weights: the exact layout

Model weights are usually reused across many calls. They can be rearranged once into an order convenient for SIMD across outputs and kept immutable. This is the prepared representation, referred to below as prepared. With `M >= L`, its shape and addressing are:

```text
packed shape = [ceil(M/C), K, C]
packed elements = ceil(M/C) * K * C
packed bytes = packed elements * sizeof(float)

tile = floor(c / C)
lane = c % C
packed[tile*K*C + i*C + lane] = W[c*K + i]
```

The last stripe is **zero-padded only along the outputs M**. K is not padded to an arbitrary SIMD size. The byte size including padding is checked before allocation and included in the budget.

Example for three outputs, K=2, and C=4:

```text
W = [a0 a1]    packed = [a0 b0 c0 0,
    [b0 b1]              a1 b1 c1 0]
    [c0 c1]
```

This is the **CPU backend's layout in main memory**. The on-disk quantization format and the general model representation are defined separately. M, K, the layout, its width, and a compatible kernel are stored alongside the data. A prepared matrix for one profile cannot be passed to another profile's function without checking compatibility: their addressing may differ.

With `M < L`, the current implementation stores exactly `M*K` elements **in row-major order** and associates them with the ordinary dot-product kernel. This avoids inflating a narrow matrix through zero-padding to a full stripe. For example, M=1/K=1024 requires 4 KiB; an AVX-512 stripe of 64 outputs would take 256 KiB.

The prepared object owns a copy of the weights; the original matrix can be freed or modified. The current allocator aligns this object to 256 bytes. Ordinary kernel inputs, however, need only valid FP32 alignment: SIMD loads accept addresses without additional alignment. The owner's 256-byte alignment is an implementation property, not a requirement of the operation itself.

Multiple calls and sessions can read one prepared object. In a graph, weights used only by linear operations are packed during preparation. If one table serves both embedding and the output projection, it retains a single row-major representation when packing would require an extra permanent copy. The operation accesses existing data and does not create a new owner on every step.

## 8. GEMV with prepared weights

Preparation changes the order in which weights are read while preserving the same formula. SIMD operates across outputs here: at each K step, a vector of adjacent weights is loaded, and the corresponding X value is broadcast to every lane. Each SIMD lane accumulates its own Y, so the lanes do not need to be summed together.

One microkernel can process several complete storage stripes at once. This provides the processor with enough independent sums for parallel computation:

| Profile from the measured build | Storage stripe C | Outputs in a GEMV group | Stripes per group | K chains per output |
|---|---:|---:|---:|---:|
| x86 baseline | 8 | 16 | 2 | 2 |
| AVX2 | 16 | 32 | 2 | 2 |
| AVX-512 | 64 | 64 | 1 | 1 |

In general, the GEMV group width is `G = L*dot_columns`. The choice follows register geometry, not the CPU name.

With two stripes, even and odd K values feed independent accumulators. After the paired loop, the chains are added together; if K is odd, the last element is computed separately. A single wide stripe keeps one chain: doubling the chains did not give an overall improvement in the tested AVX-512 profile.

The second stripe starts `C*K` elements after the beginning of the first. Its vector for the same K does **not** immediately follow the first stripe's vector. For example, in AVX2, stripe 0 is at `packed[0 .. 16*K)`, and stripe 1 is at `packed[16*K .. 32*K)`. Each has its own contiguous traversal along K. If the two stripes are incorrectly assumed to interleave within one tile of width 32, the kernel reads the wrong weights.

### Output remainders

Full groups G are written directly to Y. An incomplete group must not read nonexistent stripes merely for SIMD convenience.

Current rules:

1. If the actually allocated stripes of the final remainder fill a complete GEMV group, the common group kernel is allowed. The code requires both `remainder > G-C` and `K >= 2*(G/C)` when `G>C`; this group's incomplete output is written through a temporary array.
2. Otherwise, the existing full stripes C and one final padded stripe are processed.
3. A partial stripe uses a temporary output and bias padded with zeros beyond the real M. Only existing outputs are copied into the final Y.
4. A single remaining stripe can retain two K chains without reading a second stripe. This variant is enabled only when `K >= 2*(G/C)`; it does not pay off for very short K.

These conditions allow use of zero-padding within the allocated packed buffer, but prohibit reading past its end or writing past the end of the user's Y. Calling code does not have to allocate a hidden reserve of elements “for SIMD.”

## 9. GEMM with prepared weights

On repeated calls, the weights are already in the required order. Repacking is unnecessary, but blocking still helps reuse loaded data across several input rows.

For sufficiently large N (`N >= packing_rows`) and a full output stripe, the same principle of row blocks and K blocks is used as in ordinary GEMM:

```text
for a row block rb:
    for a full output stripe c:
        for a K block starting at kb:
            take the already packed weights at address c*K + kb*C
            update the compact sums array
            add bias only on the last K block
        transfer the completed sums to Y with ld_y
```

The main wide register tile processes six rows in the current profiles. Remaining rows are handled in groups of 4, 2, and 1. The partial-sum array `sums` is limited to 64 KiB; a temporary weight panel is no longer needed. For AVX2, the main `sums` array occupies 16 KiB.

For small N, a direct traversal over all K is performed with groups of up to four rows. A reduced group is also used for the final incomplete output stripe: results are written to a temporary tile, then only the real width is copied.

A six-row tile helped blocked GEMM but worsened timings for some small batches and remainders. It therefore cannot be applied mechanically to every path. The register tile size and the temporary-sum block size are selected separately.

Blocked GEMM requires a large temporary array on the stack, so it is placed in a separate function. Short GEMV calls consequently avoid spending time setting up that array. In the current code, the beginning of the prepared function's machine code is also aligned to 32 bytes; this refers to the function, not its input data. This is a preserved code-layout setting, not a portable guarantee of acceleration.

## 10. Streaming GEMV

Ordinary row-major GEMV reads several distant rows of W at once, reusing X. This is useful when weights reside in CPU cache. For a large matrix that must be read from main memory, several independent read streams can make data delivery more difficult.

The streaming variant traverses rows of W **sequentially, one output at a time**. Within a row, it uses up to eight independent vector accumulators to reduce dependencies between FMAs. The common generator selects the chain count as a power of two no greater than `clamp(R-2, 1, 8)`; the current x86 profiles use eight.

The name Streaming refers to sequential row reads and the size of the group computed simultaneously. The mathematics remains the same. This mode does not use a special non-temporal load instruction and does not imply bypassing the cache.

Whether weights fit in cache is estimated from the **entire matrix before it is partitioned among workers**. Otherwise, each thread would see only a small fraction of the full matrix and incorrectly conclude that there is enough room. The heuristic uses the detected shared-cache size; Adaptive can test another mode using actual execution time.

For prepared weights, the sequential row-major arithmetic flag is disabled: the adaptive plan has `streaming=false`. The worker count and job geometry remain separate decisions, while the arithmetic matches the prepared layout. Transposing the weights back on every call is not allowed.

## 11. What the handwritten SIMD kernels did

In the handwritten path, the developer explicitly selects x86 intrinsics; a separate scalar loop also exists. Many of the same techniques are used here, but the parameters and matrix traversal differ from the common generator. The latest measurements of the generated path with prepared weights do not describe these handwritten kernels.

| Path | Structure |
|---|---|
| GEMV | Four outputs reuse a vector of X; AVX2 has two chains per output, AVX-512 has four; then the remaining full vectors and the scalar K tail |
| Small batch | Up to four input rows; groups of 2 outputs for AVX2 and 4 for AVX-512; one vector accumulator per row/output pair |
| Ordinary packed GEMM | Panels up to 128 deep, 16 wide for AVX2 and 32 for AVX-512; outer product across outputs; register tile of up to 6 and 8 rows, respectively |
| Wide AVX-512 GEMM | Stripe of 64 outputs, up to 128 K values in a panel, four rows in the main register tile; enabled for N of at least 128 with sufficient M |
| Streaming GEMV | One row of W at a time, eight independent accumulators |

The threshold for ordinary local packing in the handwritten path is N of at least 64 for AVX2 and 16 for AVX-512, with M at least two SIMD vectors wide. These thresholds differ from those of the common generator.

Ordinary handwritten packed GEMM traverses **output stripe → K panel → all input rows**. It has no separate `block_rows` block or sums array: the first K block starts from zero and writes Y, subsequent blocks read the partial result from Y, and the last adds bias. Temporary storage consists only of the panel: 8 KiB for AVX2, 16 KiB for AVX-512. Incomplete output tiles are handled by dot products; in handwritten GEMV, output remainders after groups of four are handled one at a time.

The wide AVX-512 kernel has two write modes. With suitable alignment, partial sums are written directly to Y between K panels. A compact temporary sums array followed by a copy is selected when all of `128 <= N <= 256`, `K >= 1024`, and `(ld_y % 16 != 0 || address(Y) % 64 != 0)` hold. The weight panel occupies 32 KiB, and sums occupies up to 64 KiB. The direct variant is also correct for unaligned outputs outside these bounds. This works around losses from access to an inconvenient output address; it is not an alignment requirement for public tensors.

Techniques worth preserving from the handwritten path include several independent accumulation chains, reuse of X and W in registers, switching to outer products after packing, bounded temporary storage, a separate short function for GEMV, and handling full SIMD vectors before the scalar remainder. Specific intrinsics and thresholds are worth retaining when their benefit is confirmed on supported processors.

## 12. Parallel execution

So far, the discussion has concerned computation on one thread. To use several CPU cores, the result matrix is divided into independent regions. An arithmetic kernel receives one such region and does not create threads itself: a pool created in advance distributes the work. This makes it possible to measure microkernel speed and full-call time separately.

The main partitioning principle is **independent rectangles of Y along rows and output features**. Each worker completely computes the output numbers assigned to it. K is not split between workers, so no shared FP32 reduction, atomic addition of outputs, or second pass to combine partial results from different threads is required.

Wide M is first divided into output stripes so that each worker retains weight reuse across input rows. Narrow M additionally requires splitting N. Prepared-stripe boundaries align with C; the final boundary may coincide with the real M. For narrow row-major prepared weights, all of M forms one stripe, and parallelism is sought along N.

For example, a worker with columns `[c0,c1)` and rows `[r0,r1)` receives:

```text
x_base = X + r0*K
w_base = W_or_packed + c0*K  # for packed, c0 must be on a C boundary
y_base = Y + r0*M + c0
b_base = bias + c0, if bias is present
N_local = r1-r0
M_local = c1-c0
ld_y = M                     # original width, not M_local
```

Overly narrow neighboring output stripes can fall within one cache line. Threads write different elements but still interfere with one another through false sharing. Job-size selection therefore accounts for output-stripe width and the possibility of dividing work by rows.

The exact partitioning rules for prepared weights follow. `P` is the maximum worker count, `U` is the output partitioning unit (width C for packed weights, or all of M for narrow row-major weights), and `T=ceil(M/U)` is the number of such tiles. With `row_split=0`, output stripes are preferred; requests 1/2/3 express a preference for more row groups:

```text
q = 2^min(row_split, 3)
if 1 <= N <= 4:
    stripes = min(max(T, 1), P)
    row_chunk = 4
    row_groups = 1
else:
    stripes = min(max(T, 1), max(floor(P/q), 1))
    row_chunk = 4 * ceil(ceil(N / ceil(P/stripes)) / 4)
    row_groups = ceil(N / row_chunk)

base = floor(T/stripes)
extra = T % stripes
first_tile(s) = s*base + min(s, extra)
end_tile(s) = first_tile(s) + base + (s < extra ? 1 : 0)

jobs = stripes * row_groups
active = min(P, jobs)
```

The first `extra` stripes receive one additional tile. Each pair of a row group and a stripe forms a separate job; the last column is clamped to the real M, and the last row to N. A request q does not guarantee exactly q resulting row groups.

The configured thread count is the pool's capacity, not a promise to use every thread. Small work may run on one thread, and a task with few tiles may use only part of the selected group. With SMT, physical cores and logical threads must not be treated as identical resources.

Resource-placement information comes from the OS: allowed CPUs, physical cores, SMT siblings, shared caches, and NUMA. Distinct physical cores are selected first, then SMT siblings are added. The final ordering of the main group accounts for NUMA, shared cache, and core; SMT siblings may be adjacent. The physical-core group is therefore stored as a separate table of worker indices and does not necessarily coincide with the first P workers of the main group. If placement is unknown or can change freely, hints about physical cores and cache must not be invented. Placing threads across NUMA nodes does not by itself place memory there.

CCD numbers, CPU ordering, and processor model must not become conditions inside the arithmetic block.

### Persistent pool and job lifetime

Threads are created in advance and reused. A mutex permits submissions to a pool only one at a time: the pool's shared slot holds one active work description. The calling thread, hereafter `caller`, publishes the job and waits for completion. It does not join the group as an additional compute worker.

A job selects a registered immutable group and the first required workers within it: an active prefix. If there are more rectangles than active workers, the worker with rank `rank` handles job numbers `rank`, `rank+active`, `rank+2*active`, and so on. Each worker immediately knows the indices of its regions; a shared atomic counter for distributing individual tiles is unnecessary.

The command encodes a group number, active-prefix size, and generation. **The generation counter is separate for each `(group, active)` pair**. Workers can skip commands for other groups/prefixes; after wrapping, a single global counter could make a new job indistinguishable from an old one to a worker that has been excluded for a long time.

A ready result alone is insufficient for a safe return from the call: workers must finish all accesses to the supplied data. The following memory-access ordering ensures this. Release, Acquire, and AcqRel denote atomic-operation ordering modes; they must not be changed without verifying the protocol.

```text
caller:  fill Job with pointers to live inputs/outputs and the callback
caller:  publish the new command with Release
worker:  observe the command with Acquire, check membership in the selected group
worker:  read Job and execute its part
worker:  finish ALL accesses to Job and the borrowed context
worker:  publish completion (AcqRel in the current protocol)
caller:  wait for every selected worker's completion with Acquire
caller:  only now replace Job, return, or free the context
```

Excluded workers do not read Job. A completion signal means that all accesses to borrowed data have ended, not merely that Y is ready. Otherwise, the calling thread could free an object while a worker is still inspecting its fields or finishing the callback.

If a worker submits a nested job to **the same pool**, it executes that job directly on its own thread. This case is detected through a thread-local pool identifier: waiting for a pool occupied by its own work would cause a deadlock. Cyclic callback dependencies between different pools are not supported. The current Rust pool catches panics; all workers reading the context finish first, then the error is passed to the calling thread. C must preserve this ordering of waiting for completion and freeing memory on error, although the Rust panic mechanism itself need not be ported.

### Waiting and waking

After a small GEMV is submitted, the result may be ready very quickly. A short busy wait can receive it without an OS context switch. If work takes longer, the thread sleeps to avoid occupying the processor: this combines spinning followed by parking.

In the old implementation, the caller uses a shared budget for busy-waiting on the entire group: 256 spin iterations for `Short`, 4096 for `Long`; Linear selects `Long`. The budget is set to zero if the active worker count is at least the number of available CPUs, or if the caller is running on the CPU of a selected pinned worker. Between jobs, a worker normally performs 4096 spin iterations, but immediately parks after a command that excludes it or after working on the caller's CPU. These are settings of the old implementation; they may be unsuitable for another CPU.

The transition to sleep is coordinated with work submission. Otherwise, a new job could appear between the last check and park, leaving a worker asleep without noticing it. The sequence is:

1. The worker registers its bit in the waiting bitmap through AcqRel.
2. It rereads the command through Acquire; if work has already arrived, it does not sleep.
3. After publishing the command, the caller clears only the selected workers' bits through AcqRel and calls `unpark` for those registered. Excluded workers' bits are preserved.

A similar exchange is required when the caller sleeps while waiting for the result. The low bit of completion represents a waiting caller: it sets the bit through AcqRel and checks the previous value. The worker publishes completion through an AcqRel swap and calls `unpark` if the bit was set. In the current implementation, epoch and completion are aligned to 128 bytes to reduce cache-line traffic between independent fields.

### Fixed and Adaptive

Worker count also affects time: dispatching a small task to a large group can cost more than computing it sequentially. The existing implementation provides two ways to select a schedule.

`Fixed` decides using predetermined rules and available resources. In particular, for prepared weights with a pool of more than one thread, work of size `N*M*K >= 262144` selects the physical-core group if known, otherwise the main group. Smaller work selects a sequential plan. This is a historical threshold on the multiplication count, not a universal CPU characteristic. Sequential execution does not always mean the caller: a single-thread pool with mandatory affinity executes work on its pinned worker.

`Adaptive` observes the duration of real calls and selects a schedule for the shape `(N,M,K)`. Its initial candidate **does not match Fixed for every shape**: when parallelism is available, it starts with the physical-core group or the main group, then considers half the initial group, the entire main group, and a sequential plan. Duplicate variants are removed. The policy does not change the mathematical rules of the operation or the CPU instruction set. For prepared calls, training results also relate to the particular stripe width. The executor keeps observations for ordinary and prepared weights separately.

The current version has up to eight candidates: different worker counts and available groups, GEMV with ordinary weights using ordinary or sequential reads, and, for prepared GEMM, row-splitting variants requesting 2, 4, or 8 groups. If rounding produces identical partitions, only one is measured. For N from 1 to 4, row splitting produces no additional groups; output stripes remain.

Each candidate under investigation receives one warmup call and three measured calls. The sequential candidate may be skipped based on the cost of handing work to the pool; the reason is described below. If only one plan remains after deduplication, it is published immediately without measurements. When comparing several plans, the initial schedule is checked again at the end: one warmup and three measurements. The upper bound is **36 successful calls without contention for training**, provided the entry is not evicted. Observations are collected on ordinary user calls: each computes the result **exactly once**, without repeating the entire matrix operation to compare alternatives.

The winner is selected conservatively: the median of a candidate's three observations must be lower than the best time for the initial schedule observed before or after the search. A tie retains the initial variant. This protects against a single lucky candidate measurement and some frequency transitions, but does not guarantee the right choice in the presence of noise.

The cost of handing an empty job to the pool is measured when the executor is created. It is used to limit exploration of single-thread execution: large tasks should not be assigned entirely to one thread solely for autotuning.

Each tuner keeps an observation cache of up to 32 entries. A ready hint is read atomically; training uses `try_lock`. If the entry is busy, the call follows the safe initial plan and does not wait for training to finish. A failed call does not count as an observation. Evicting an entry can restart training.

The cache stores **a resource hint, not memory addresses or sizes**. Even in the event of a collision or stale observation, job boundaries are derived afresh from the validated current shape. The diagnostic `max_workers` is an upper bound, not a count of threads actually busy.

Known limitation: short calls are sensitive to OS noise, temperature, frequency, and cache state. In earlier runs, Adaptive selected different groups for the same shape. Kernel comparisons therefore need both a fixed schedule and a measurement of the real adaptive path after training; one result cannot be substituted for the other.

## 13. What must survive a rewrite

The following rules connect arithmetic, memory, and threads. Preserving only the multiplication formula is insufficient for a correct port.

- Preparation binds a layout to its kernel and accounts for padding in advance. Repeated calls neither repack constant weights nor allocate temporary heap storage.
- Lifetimes are explicitly defined for original data, prepared weights, temporary storage, and output. All workers finish using borrowed pointers before the call returns.
- Output rectangles do not overlap; the Y stride remains that of the full matrix. Bias is added exactly once, not on every K block.
- The first block completely initializes its sums. Uninitialized temporary storage or old NaN values in Y must not be read.
- Remainders along N, M, and K are checked separately. Weight padding does not authorize overwriting padding that the output does not have.
- A portable baseline and forced ISA selection remain available for comparisons. AVX-512 support does not prove that it is faster for every shape.
- ARM support in the common generator is checked through cross-compilation and correctness; real ARM performance cannot be inferred from QEMU or x86 measurements.
- The experimental flag remains explicitly separate from the accepted path. A favorable shape or a single best measurement is insufficient reason to replace the common algorithm.

## 14. How to validate the reconstructed implementation

First, the reconstructed kernel is compared against an independent reference and checked for out-of-bounds memory accesses. Only then does it make sense to evaluate speed. The minimum set of cases is:

| Check | What it catches |
|---|---|
| N=0, N=1, groups of 2/4/6/8 and neighboring sizes | Empty calls, GEMV, register-tile tails |
| M=1 and M below the SIMD width | Unnecessary padding of narrow weights, incorrect prepared-layout selection |
| M=C-1, C, C+1, G-1, G, G+1 | Stripe and group remainders in prepared GEMV |
| K=1,2,3; odd K; sizes around L and L*chains | Missing contributions, repeated tails, reading an extra vector |
| Sizes around depth and block_rows | Zeroing/continuation of sums and adding bias once |
| Unaligned X/W/Y, ld_y greater than the local width | Alignment assumptions and corruption of output gaps |
| Bias absent and present | Null pointers and correct bias indices |
| Y prefilled with NaN | Accidental accumulation into the old result |
| NaN/Inf in only one row of W | Mixing independent outputs during reduction |
| Canary values and a protected page immediately after the input | Out-of-bounds writes and reads; canaries alone are insufficient for reads |
| Modification/removal of the original W after preparation | Independence of the owning prepared snapshot |
| Invalid dimensions, overflow, incompatible ISA/layout | Rejection before changing Y and before incorrect allocation |
| 1, 3, the physical core count, and an SMT pool | Uneven partitioning, incomplete groups, and the completion protocol |
| Multiple repeated and concurrent calls | Pool state, lifetimes, and absence of memory growth |

Repeated calls must also be checked separately for heap allocations, including during Adaptive training and calls from a graph with borrowed weights. Profiling accounts for both the FMA loop and the surrounding work: loads, packing, result writes, worker wakeups, and graph checks take time too.

## 15. The measured starting point

The data below describes a **Ryzen 9 9950X, FP32, a limit of 16 threads**, warm repeated calls, and a build of the common LLVM path dated 2026-11-08. It is neither a promise for another CPU nor a measurement of full model inference. Preparation of constant weights and executor creation are excluded from repeated-call time.

| Task N×M×K | Ordinary weights, TFLOPS | Prepared, TFLOPS | Reference point |
|---|---:|---:|---|
| 256×4096×4096 | 4,2200 | 4,3808 | Arithmetic: 4,8230 TFLOPS at frequencies derived from counters |
| 1×1024×1024 | 0,9411 | 1,0754 | Weight delivery from L2: a conditional 2,6890 TFLOPS |
| 1×8192×16384 | 0,0359 | 0,0351 | The physical RAM ceiling was not established; a separate read test yielded an equivalent 0,0341 TFLOPS |
| 4×4096×4096 | 1,4512 | 1,9633 | Large variation between processes; arithmetic peak alone is insufficient |

The arithmetic reference point is calculated as `64 FP32 FLOP/cycle × sum of the frequencies of 16 physical cores`. The cached-GEMV reference point is `64 bytes/cycle from L2 × 2 FLOP / 4 bytes per weight × sum of frequencies`. These are conditional models: data delivery, synchronization, and actual residency are not established by one such ratio. SMT does not double the execution units. A control RAM read must not be declared a physical ceiling, especially when real GEMV is faster than it.

On the main GEMM, AOCL prepared achieved 4,3095 TFLOPS, and LIBXSMM with the measurement adapter achieved 4,2912. On narrow N=4, LIBXSMM achieved 2,0454 versus our 1,9633, with heavily overlapping ranges. These measurements support retaining the common generator, but do not prove that all its parameters are optimal.

Reproducibility requires preserving the exact build/sources, dtype and shapes, layout and its size, thread count and placement, selected schedule, warmup, preparation cost separately, raw measurements, and numerical checks. Compare a full call with a full call; a loop with hundreds of repetitions inside a single pool job measures a different mode.

## 16. Where the source components were located

These paths are pointers into the old checkout or archive, not required files in the future structure:

| Old file | What it preserved |
|---|---|
| `crates/lumen-cpu/codegen/linear.rs` | Geometry, generation of dot/outer/transpose LLVM IR and the C driver |
| `crates/lumen-cpu/codegen/linear.c` | Selection of ordinary GEMV, batch, packed GEMM, and streaming |
| `crates/lumen-cpu/codegen/packing.c` | Local panels of ordinary weights and partial-sum blocks |
| `crates/lumen-cpu/codegen/prepared.c` | Prepared GEMV/GEMM and all kinds of tails |
| `crates/lumen-cpu/codegen/dot.rs` | Experimental completion of the vector dot tail |
| `crates/lumen-cpu/codegen/capabilities.cpp`, `build.rs` | Obtaining target information and AOT compilation |
| `crates/lumen-cpu/src/linear/compiled.rs` | Profile compatibility, machine-code calls, job geometry |
| `crates/lumen-cpu/src/linear/prepared.rs` | Owning and borrowed prepared weights |
| `crates/lumen-tensor/src/dense_layout.rs` | Size formulas and packing of the physical layout |
| `crates/lumen-cpu/src/linear.rs`, `linear/wide.rs`, `linear/streaming.rs` | Handwritten SIMD variants and partitioning of ordinary Linear |
| `crates/lumen-cpu/src/linear/policy.rs`, `executor.rs`, `threading/` | Group selection, schedule training, and the persistent pool |
| `crates/lumen-ref/src/linear.rs` | Independent FP64 reference |
| `crates/lumen-cpu/tests/`, `linear/compiled/tail_tests.rs` | Shapes, tails, ISA, guard pages, prepared weights, and parallelism |
| `crates/lumen-exec/src/prepare.rs`, `crates/lumen-backend/src/validation.rs` | Graph weight preparation, memory, and validation |

Additional materials, if retained with the documentation: library comparison and raw measurements (`../benchmarks/linear-libraries-2026-11-08/README.md`), source snapshot of the measured version (`../benchmarks/linear-libraries-2026-11-08/lumen-source.tar.gz`), source hashes and machine parameters (`../benchmarks/linear-libraries-2026-11-08/manifest.json`), experiment history (`../benchmarks/linear-night-2026-11-08/README.md`). When the code is removed, this Markdown remains a standalone description, while the archive allows the exact details of the previous implementation to be recovered if needed.
