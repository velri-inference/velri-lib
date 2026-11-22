#ifndef VELRI_TEST_GUARD_BUFFER_H
#define VELRI_TEST_GUARD_BUFFER_H

#include <stddef.h>

enum guard_edge { GUARD_NONE, GUARD_BEFORE, GUARD_AFTER };

typedef struct guard_buffer {
    void *data;
    void *mapping;
    size_t mapping_bytes;
} guard_buffer;

/* A zero-filled buffer of nonzero size. With GUARD_BEFORE or GUARD_AFTER, the
 * selected end touches an inaccessible page. With GUARD_AFTER, the byte count
 * must be divisible by the required element alignment. */
guard_buffer guard_buffer_alloc(size_t bytes, enum guard_edge edge);
void guard_buffer_free(guard_buffer buf);

#endif
