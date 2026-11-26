#ifndef VELRI_CORE_UTIL_INTERNAL_H
#define VELRI_CORE_UTIL_INTERNAL_H

#include "velri_lib/types.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Lightweight, dependency-free internal utilities shared across velri-lib modules
 * (overflow-checked arithmetic, alignment, strict-aliasing-safe dynamic vectors, refcounts).
 */

velri_status velri_shape_byte_size(velri_dtype dtype, uint32_t rank, const size_t *dims,
                                   size_t *out_bytes);

static inline bool velri_is_pow2(size_t x) { return x != 0U && (x & (x - 1U)) == 0U; }

static inline bool velri_mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (a != 0U && b > SIZE_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static inline bool velri_add_overflow_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) {
        return true;
    }
    *out = a + b;
    return false;
}

static inline size_t velri_sat_add_size(size_t a, size_t b) {
    return (a > SIZE_MAX - b) ? SIZE_MAX : (a + b);
}

static inline bool velri_align_up_overflow(size_t n, size_t align, size_t *out) {
    if (align <= 1U) {
        *out = n;
        return false;
    }
    if (!velri_is_pow2(align)) {
        return true;
    }
    size_t mask = align - 1U;
    if (n > SIZE_MAX - mask) {
        return true;
    }
    *out = (n + mask) & ~mask;
    return false;
}

static inline size_t velri_align_up(size_t n, size_t align) {
    size_t out = 0U;
    return velri_align_up_overflow(n, align, &out) ? SIZE_MAX : out;
}

static inline uint32_t velri_div_ceil_u32(uint32_t a, uint32_t b) {
    return (a == 0U) ? 0U : (1U + (a - 1U) / b);
}

static inline velri_status velri_calc_alloc_bytes(size_t bytes, size_t align, size_t min_align,
                                                  size_t *out_align, size_t *out_bytes) {
    size_t eff_align = (align == 0U) ? min_align : align;
    if (!velri_is_pow2(eff_align)) {
        return VELRI_ERR_INVALID;
    }
    if (eff_align < min_align) {
        eff_align = min_align;
    }
    size_t req = bytes ? bytes : eff_align;
    size_t alloc_sz = 0U;
    if (velri_align_up_overflow(req, eff_align, &alloc_sz)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_align = eff_align;
    *out_bytes = alloc_sz;
    return VELRI_OK;
}

/* Strict-aliasing-safe dynamic array growth (never casts &typed_ptr to void **).
 * Returns the (possibly moved) buffer on success, or nullptr on failure (leaving `ptr` intact). */
static inline void *velri_vec_grow(void *ptr, uint32_t *cap, uint32_t need, size_t elem_size) {
    if (need <= *cap) {
        return ptr;
    }
    uint32_t next = (*cap == 0U) ? 16U : *cap;
    while (next < need) {
        if (next > UINT32_MAX / 2U) {
            next = need;
            break;
        }
        next *= 2U;
    }
    size_t bytes = 0U;
    if (velri_mul_overflow_size((size_t)next, elem_size, &bytes)) {
        return nullptr;
    }
    void *p = realloc(ptr, bytes);
    if (!p) {
        return nullptr;
    }
    *cap = next;
    return p;
}

/* Portable strict-ISO-C23 reserve helper: checks u32 overflow and preserves `old_ptr` on OOM. */
static inline bool velri_vec_ensure(void *old_ptr, void **out_new_ptr, uint32_t *cap,
                                    uint32_t count, uint32_t add, size_t elem_size) {
    if (count > UINT32_MAX - add) {
        return false;
    }
    uint32_t need = count + add;
    if (need <= *cap) {
        *out_new_ptr = old_ptr;
        return true;
    }
    void *p = velri_vec_grow(old_ptr, cap, need, elem_size);
    if (!p) {
        return false;
    }
    *out_new_ptr = p;
    return true;
}

static inline void velri_ref_inc(_Atomic uint32_t *rc) {
    atomic_fetch_add_explicit(rc, 1U, memory_order_relaxed);
}

/* Returns true when the caller decremented the final reference (count dropped 1 -> 0). */
static inline bool velri_ref_dec(_Atomic uint32_t *rc) {
    return atomic_fetch_sub_explicit(rc, 1U, memory_order_acq_rel) == 1U;
}

#endif /* VELRI_CORE_UTIL_INTERNAL_H */
