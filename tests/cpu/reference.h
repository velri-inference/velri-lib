#ifndef VELRI_TEST_REFERENCE_H
#define VELRI_TEST_REFERENCE_H
#include <stddef.h>

/* Independent row-major oracle, no production helpers or packed layout.
 * Same sequential FP64 arithmetic as the archived Rust ref_linear_f32;
 * canonical NaN is the sole explicit refinement of its unspecified NaN bits.
 */
void reference_linear(const float *x, const float *w, const float *bias, float *y, size_t rows,
                      size_t m, size_t k, size_t xs, size_t ws, size_t ys);
#endif
