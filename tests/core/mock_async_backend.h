#ifndef VELRI_TEST_MOCK_ASYNC_BACKEND_H
#define VELRI_TEST_MOCK_ASYNC_BACKEND_H

#include "velri_lib/backend.h"
#include <stdatomic.h>

/*
 * Controllable asynchronous mock backend (Contract v2) for core runtime tests.
 * - Exposes one non-host device (`mock_async:0`, isolated device memory).
 * - Supports ONLY `VELRI_OP_VEC_AXPBY` (verifies strict no silent CPU fallback).
 * - Executes work on a real background OS thread with configurable delay,
 *   pre-accept rejection, post-accept failure, and partial-accept error injection.
 */

typedef struct {
    atomic_uint delay_ms;
    atomic_bool fail_pre_accept;
    atomic_bool fail_post_accept;
    atomic_bool fail_partial_submit;
    atomic_int in_flight;
    atomic_uint total_submitted;
    atomic_uint total_completed;
    atomic_uint total_uploads;
    atomic_uint total_downloads;
} velri_mock_async_ctrl;

const velri_backend_vtable *velri_mock_async_backend_vtable(void);
velri_mock_async_ctrl *velri_mock_async_control(void);
void velri_mock_async_reset_ctrl(void);

#endif /* VELRI_TEST_MOCK_ASYNC_BACKEND_H */
