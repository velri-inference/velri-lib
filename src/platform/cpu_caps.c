#include "os.h"
#include "velri_lib/types.h"
#include <stdatomic.h>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__x86_64__) || defined(_M_X64)
static uint64_t read_xcr0(void) {
#if defined(_MSC_VER)
    return (uint64_t)_xgetbv(0);
#else
    uint32_t eax = 0, edx = 0;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#endif
}

static void detect_x86_features(velri_cpu_info *info) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    unsigned int max_leaf = 0;
#if defined(_MSC_VER)
    int r0[4] = {0};
    __cpuid(r0, 0);
    max_leaf = (unsigned int)r0[0];
    if (max_leaf < 1U) {
        return;
    }
    int r1[4] = {0};
    __cpuid(r1, 1);
    ecx = (unsigned int)r1[2];
    (void)eax; (void)ebx; (void)edx;
#else
    max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf < 1U || !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return;
    }
#endif
    bool osxsave = (ecx & (1U << 27)) != 0U;
    uint64_t xcr0 = osxsave ? read_xcr0() : 0U;
    bool ymm_os = (xcr0 & 0x6ULL) == 0x6ULL;
    bool zmm_os = (xcr0 & 0xE6ULL) == 0xE6ULL;

    if (ymm_os) {
        info->has_fma = (ecx & (1U << 12)) != 0U;
        info->has_f16c = (ecx & (1U << 29)) != 0U;
    }
    if (max_leaf < 7U) {
        return;
    }

    unsigned int max_subleaf = 0;
#if defined(_MSC_VER)
    int r7[4] = {0};
    __cpuidex(r7, 7, 0);
    max_subleaf = (unsigned int)r7[0];
    ebx = (unsigned int)r7[1];
    ecx = (unsigned int)r7[2];
#else
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return;
    }
    max_subleaf = eax;
#endif
    if (ymm_os) {
        info->has_avx2 = (ebx & (1U << 5)) != 0U;
    }
    if (zmm_os) {
        info->has_avx512f = (ebx & (1U << 16)) != 0U;
        info->has_avx512dq = (ebx & (1U << 17)) != 0U;
        info->has_avx512bw = (ebx & (1U << 30)) != 0U;
        info->has_avx512vl = (ebx & (1U << 31)) != 0U;
        info->has_avx512_vnni = (ecx & (1U << 11)) != 0U;
    }

    if (max_subleaf < 1U) {
        return;
    }
    unsigned int a1 = 0, b1 = 0, c1 = 0, d1 = 0;
#if defined(_MSC_VER)
    int r71[4] = {0};
    __cpuidex(r71, 7, 1);
    a1 = (unsigned int)r71[0];
    (void)b1; (void)c1; (void)d1;
#else
    if (!__get_cpuid_count(7, 1, &a1, &b1, &c1, &d1)) {
        return;
    }
#endif
    if (zmm_os && info->has_avx512f && info->has_avx512bw && info->has_avx512vl) {
        info->has_avx512_bf16 = (a1 & (1U << 5)) != 0U;
    }
}
#endif

static velri_cpu_info g_info;
static atomic_int g_init_state = 0; /* 0 = uninit, 1 = busy, 2 = ready */

static void init_cpu_info_once(void) {
    if (atomic_load_explicit(&g_init_state, memory_order_acquire) == 2) {
        return;
    }
    int expected = 0;
    if (!atomic_compare_exchange_strong_explicit(&g_init_state, &expected, 1, memory_order_acq_rel,
                                                 memory_order_acquire)) {
        while (atomic_load_explicit(&g_init_state, memory_order_acquire) != 2) {
            velri_os_cpu_relax();
        }
        return;
    }

    velri_cpu_info info = {0};
    info.hardware_threads = velri_os_hardware_threads();
#if defined(__x86_64__) || defined(_M_X64)
    detect_x86_features(&info);
#endif

    g_info = info;
    atomic_store_explicit(&g_init_state, 2, memory_order_release);
}

const velri_cpu_info *velri_get_cpu_info(void) {
    init_cpu_info_once();
    return &g_info;
}

bool velri_cpu_variant_supported(velri_cpu_variant variant) {
#if VELRI_X86_VARIANTS
    const velri_cpu_info *c = velri_get_cpu_info();
#endif
    switch (variant) {
    case VELRI_CPU_AUTO:
    case VELRI_CPU_BASE:
        return true;
#if VELRI_X86_VARIANTS
    case VELRI_CPU_AVX2:
        return c->has_avx2 && c->has_fma && c->has_f16c;
    case VELRI_CPU_AVX512:
        return c->has_avx2 && c->has_avx512f && c->has_avx512bw && c->has_avx512dq &&
               c->has_avx512vl && c->has_fma && c->has_f16c;
#endif
    default:
        return false;
    }
}

velri_cpu_variant velri_cpu_resolve_variant(velri_cpu_variant variant) {
    if (variant != VELRI_CPU_AUTO) {
        return variant;
    }
#if VELRI_X86_VARIANTS
    if (velri_cpu_variant_supported(VELRI_CPU_AVX512)) {
        return VELRI_CPU_AVX512;
    }
    if (velri_cpu_variant_supported(VELRI_CPU_AVX2)) {
        return VELRI_CPU_AVX2;
    }
#endif
    return VELRI_CPU_BASE;
}
