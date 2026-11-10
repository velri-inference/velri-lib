# Frozen reference result

`rust-linear-v1.bin` was produced on 10 November 2026 by the **unmodified**
archived `lumen-ref::ref_linear_f32`, before the new C kernels were added.
It is test data, not a benchmark report. Tests read it; builds never regenerate it.

SHA-256: `2c154d7fe1d7169d3da157f10ceaa66759f355fa1bc8dfb4fbdedcbe0ad3eff5`.

The source archive had SHA-256
`9227effc0eb90b7ab03e62fdca3c18aa6b292b5cd347f2834030876d6364fe1d`;
its `crates/lumen-ref/src/linear.rs` had SHA-256
`ced7536398448b21338457627a54c494f60d0eb1736eb08a6bee869899a874d6`.
The archived formula accumulates FP64 products sequentially from `-0.0`, adds
bias in FP64 after the dot product, and rounds once to FP32.

File layout, all integers and FP32 bit patterns little-endian:

1. Four bytes `VRF1`.
2. Three unsigned 64-bit dimensions: rows=3, M=33, K=257.
3. Row-major FP32 arrays X[3,257], W[33,257], bias[33], expected Y[3,33].

Both the independent C reference and every available optimized variant must
match every stored output bit. The C reference uses ordinary W rows and does
not use production packing or arithmetic helpers. Additional tests cover
random finite bit patterns, cancellation, signed zeros, subnormals, overflow,
NaNs, infinities, strides, tails, reuse and memory bounds.

The fixture has finite inputs. Rust does not specify stable NaN result payloads;
the new C contract explicitly canonicalizes NaN outputs to `0x7fc00000`.
This refinement is separate from finite-number equality and is **not** evidence
that the historical FP32 fast kernels matched the Rust reference bit for bit.
