#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include "guard_buffer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

static void guard_fail(const char *reason) {
    fprintf(stderr, "guard_buffer: %s\n", reason);
    exit(EXIT_FAILURE);
}

guard_buffer guard_buffer_alloc(size_t bytes, enum guard_edge edge) {
    if (bytes == 0 || (edge != GUARD_NONE && edge != GUARD_BEFORE && edge != GUARD_AFTER)) {
        guard_fail("invalid buffer size or edge");
    }
    if (edge == GUARD_NONE) {
        void *data = calloc(1, bytes);
        if (!data) {
            guard_fail("calloc failed");
        }
        return (guard_buffer){.data = data};
    }
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    size_t page = info.dwPageSize;
#else
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        guard_fail("sysconf(_SC_PAGESIZE) failed");
    }
    size_t page = (size_t)page_size;
#endif
    size_t pages = 1 + (bytes - 1) / page;
    size_t max_pages = SIZE_MAX / page;
    if (pages > max_pages || max_pages - pages < 2) {
        guard_fail("mapping size overflow");
    }
    size_t total = (pages + 2) * page;
#ifdef _WIN32
    unsigned char *mapping = VirtualAlloc(nullptr, total, MEM_RESERVE, PAGE_NOACCESS);
    if (!mapping || !VirtualAlloc(mapping + page, pages * page, MEM_COMMIT, PAGE_READWRITE)) {
        guard_fail("VirtualAlloc failed");
    }
#else
    unsigned char *mapping = mmap(nullptr, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED || mprotect(mapping + page, pages * page, PROT_READ | PROT_WRITE)) {
        guard_fail("mmap or mprotect failed");
    }
#endif
    void *data = edge == GUARD_BEFORE ? mapping + page : mapping + total - page - bytes;
    return (guard_buffer){.data = data, .mapping = mapping, .mapping_bytes = total};
}

void guard_buffer_free(guard_buffer buf) {
    if (!buf.mapping) {
        free(buf.data);
        return;
    }
#ifdef _WIN32
    if (!VirtualFree(buf.mapping, 0, MEM_RELEASE)) {
        guard_fail("VirtualFree failed");
    }
#else
    if (munmap(buf.mapping, buf.mapping_bytes) != 0) {
        guard_fail("munmap failed");
    }
#endif
}
