#include "reference.h"
#include <stdint.h>
#include <string.h>

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

void reference_linear(const float *x, const float *w, const float *bias, float *y, size_t rows,
                      size_t m, size_t k, size_t xs, size_t ws, size_t ys) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < m; ++c) {
            double dot = -0.0;
            for (size_t i = 0; i < k; ++i) {
                double a = x[r * xs + i];
                double b = w[c * ws + i];
                double product = a * b;
                dot = dot + product;
            }
            double offset = bias ? (double)bias[c] : 0.0;
            float result = (float)(dot + offset);
            if (result != result) {
                const uint32_t nan = UINT32_C(0x7fc00000);
                memcpy(&result, &nan, sizeof result);
            }
            y[r * ys + c] = result;
        }
    }
}
