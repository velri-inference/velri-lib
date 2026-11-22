#include "vk_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Vulkan 1.3 Instance Loader, Physical/Logical Device Selection, Capability Probing,
 * and Device Lifecycle Management.
 */

static velri_os_lib *velri_vk_dlopen(void) {
    static const char *const k_vk_libs[] = {
#if defined(_WIN32)
        "vulkan-1.dll",
#elif defined(__APPLE__)
        "libvulkan.1.dylib",
        "libvulkan.dylib",
        "libMoltenVK.dylib",
#else
        "libvulkan.so.1",
        "libvulkan.so",
#endif
        nullptr,
    };
    return velri_os_dlopen(k_vk_libs);
}

static bool has_ext(const VkExtensionProperties *exts, uint32_t count, const char *name) {
    for (uint32_t i = 0U; i < count; ++i) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

static void vk_inst_destroy(vk_inst_loader *L) {
    if (L->instance && L->vkDestroyInstance) {
        L->vkDestroyInstance(L->instance, nullptr);
    }
    velri_os_dlclose(L->dl);
    memset(L, 0, sizeof(*L));
}

static velri_status vk_inst_init(vk_inst_loader *L) {
    memset(L, 0, sizeof(*L));
    L->dl = velri_vk_dlopen();
    if (!L->dl) {
        return VELRI_ERR_UNSUPPORTED;
    }
    void *sym = velri_os_dlsym(L->dl, "vkGetInstanceProcAddr");
    if (!sym) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }
    memcpy(&L->vkGetInstanceProcAddr, &sym, sizeof(L->vkGetInstanceProcAddr));

    PFN_vkVoidFunction p_ci = L->vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    memcpy(&L->vkCreateInstance, &p_ci, sizeof(L->vkCreateInstance));
    if (!L->vkCreateInstance) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "velri-lib",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Velri",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (L->vkCreateInstance(&ci, nullptr, &L->instance) != VK_SUCCESS) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }

    bool ok = true;
#define LOAD_REQ_INST(NAME)                                                      \
    do {                                                                         \
        PFN_vkVoidFunction _p = L->vkGetInstanceProcAddr(L->instance, #NAME);    \
        memcpy(&L->NAME, &_p, sizeof(L->NAME));                                  \
        if (!L->NAME) ok = false;                                                \
    } while (0);
    VK_INST_FUNCS(LOAD_REQ_INST)
#undef LOAD_REQ_INST
    if (!ok) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }

    PFN_vkVoidFunction p_cm =
        L->vkGetInstanceProcAddr(L->instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    memcpy(&L->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, &p_cm,
           sizeof(L->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR));
    return VELRI_OK;
}

typedef struct {
    VkPhysicalDevice phys;
    uint32_t queue_family;
    bool is_discrete;
} vk_eligible_phys;

static bool probe_phys_device(const vk_inst_loader *L, VkPhysicalDevice phys, uint32_t *out_qf,
                              bool *out_discrete) {
    VkPhysicalDeviceSubgroupProperties sg = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &sg,
    };
    L->vkGetPhysicalDeviceProperties2(phys, &p2);
    if (p2.properties.apiVersion < VK_API_VERSION_1_3) {
        return false;
    }
    if (p2.properties.limits.maxComputeWorkGroupInvocations < 256U) {
        return false;
    }
    VkSubgroupFeatureFlags req_sg = VK_SUBGROUP_FEATURE_BASIC_BIT |
                                    VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                    VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if (sg.subgroupSize < 4U || sg.subgroupSize > 128U || !velri_is_pow2(sg.subgroupSize) ||
        !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
        (sg.supportedOperations & req_sg) != req_sg) {
        return false;
    }

    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f12,
    };
    L->vkGetPhysicalDeviceFeatures2(phys, &f2);
    /* All production shaders use 64-bit device-address arithmetic. */
    if (!f2.features.shaderInt64 || !f12.bufferDeviceAddress || !f12.vulkanMemoryModel ||
        !f12.vulkanMemoryModelDeviceScope || !f12.scalarBlockLayout) {
        return false;
    }

    uint32_t nqf = 0U;
    L->vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    if (nqf == 0U) {
        return false;
    }
    VkQueueFamilyProperties *qfs = (VkQueueFamilyProperties *)calloc(nqf, sizeof(*qfs));
    if (!qfs) {
        return false;
    }
    L->vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfs);
    uint32_t qf = UINT32_MAX;
    for (uint32_t q = 0U; q < nqf; ++q) {
        if (qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qf = q;
            if (!(qfs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                break;
            }
        }
    }
    free(qfs);
    if (qf == UINT32_MAX) {
        return false;
    }

    *out_qf = qf;
    *out_discrete = (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    return true;
}

static velri_status collect_eligible_devices(const vk_inst_loader *L, vk_eligible_phys **out_list,
                                             uint32_t *out_count) {
    *out_list = nullptr;
    *out_count = 0U;
    uint32_t ndev = 0U;
    if (L->vkEnumeratePhysicalDevices(L->instance, &ndev, nullptr) != VK_SUCCESS || ndev == 0U) {
        return VELRI_OK;
    }
    VkPhysicalDevice *raw = (VkPhysicalDevice *)calloc(ndev, sizeof(*raw));
    vk_eligible_phys *elig = (vk_eligible_phys *)calloc(ndev, sizeof(*elig));
    if (!raw || !elig) {
        free(raw);
        free(elig);
        return VELRI_ERR_NOMEM;
    }
    if (L->vkEnumeratePhysicalDevices(L->instance, &ndev, raw) != VK_SUCCESS) {
        free(raw);
        free(elig);
        return VELRI_OK;
    }

    uint32_t n = 0U;
    for (int pass = 0; pass < 2; ++pass) {
        bool want_discrete = (pass == 0);
        for (uint32_t i = 0U; i < ndev; ++i) {
            uint32_t qf = 0U;
            bool disc = false;
            if (probe_phys_device(L, raw[i], &qf, &disc) && disc == want_discrete) {
                elig[n++] = (vk_eligible_phys){
                    .phys = raw[i],
                    .queue_family = qf,
                    .is_discrete = disc,
                };
            }
        }
    }
    free(raw);
    if (n == 0U) {
        free(elig);
        return VELRI_OK;
    }
    *out_list = elig;
    *out_count = n;
    return VELRI_OK;
}

/* ---- 1. Registration ---- */

velri_status velri_vk_enumerate_devices(uint32_t *out_count) {
    if (!out_count) {
        return VELRI_ERR_INVALID;
    }
    *out_count = 0U;
    vk_inst_loader L;
    if (vk_inst_init(&L) != VELRI_OK) {
        return VELRI_OK; /* No Vulkan runtime installed => 0 devices */
    }
    vk_eligible_phys *list = nullptr;
    uint32_t count = 0U;
    velri_status st = collect_eligible_devices(&L, &list, &count);
    free(list);
    vk_inst_destroy(&L);
    if (st == VELRI_OK) {
        *out_count = count;
    }
    return st;
}

/* ---- 2. Device Helpers ---- */

static void populate_device_caps(velri_backend_device *dev,
                                 const VkPhysicalDeviceProperties2 *props2,
                                 const VkPhysicalDeviceSubgroupProperties *sg_props) {
    dev->subgroup_size = sg_props->subgroupSize;
    dev->max_wg_count[0] = props2->properties.limits.maxComputeWorkGroupCount[0];
    dev->max_wg_count[1] = props2->properties.limits.maxComputeWorkGroupCount[1];
    dev->max_wg_count[2] = props2->properties.limits.maxComputeWorkGroupCount[2];

    memset(&dev->caps, 0, sizeof(dev->caps));
    snprintf(dev->caps.backend_name, sizeof(dev->caps.backend_name), "%s", "vulkan");
    snprintf(dev->caps.device_name, sizeof(dev->caps.device_name), "%.127s",
             props2->properties.deviceName);
    dev->caps.local_index = dev->local_index;
    dev->caps.is_host_memory = false;

    bool all_dev_local_host_vis = true;
    bool has_dev_local = false;
    for (uint32_t i = 0U; i < dev->mem_props.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags pf = dev->mem_props.memoryTypes[i].propertyFlags;
        if (pf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            has_dev_local = true;
            if (!(pf & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                all_dev_local_host_vis = false;
            }
        }
    }
    dev->caps.is_uma =
        (props2->properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ||
        (has_dev_local && all_dev_local_host_vis);

    uint64_t total_vram = 0ULL;
    for (uint32_t i = 0U; i < dev->mem_props.memoryHeapCount; ++i) {
        if (dev->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            uint64_t hsz = (uint64_t)dev->mem_props.memoryHeaps[i].size;
            total_vram = (hsz > UINT64_MAX - total_vram) ? UINT64_MAX : (total_vram + hsz);
        }
    }
    dev->caps.mem_total_known = (total_vram > 0ULL);
    dev->caps.mem_total_bytes =
        (total_vram > (uint64_t)SIZE_MAX) ? SIZE_MAX : (size_t)total_vram;

    size_t min_align = 256U;
    size_t drv_align = (size_t)props2->properties.limits.minStorageBufferOffsetAlignment;
    while (min_align < drv_align && min_align < (1U << 20)) {
        min_align <<= 1U;
    }
    dev->caps.min_align_bytes = min_align;
}

static velri_status probe_features_and_coopmat(velri_backend_device *dev, const VkPhysicalDeviceLimits *limits,
                                               vk_probed_feats *out_feats) {
    uint32_t max_smem = limits->maxComputeSharedMemorySize;
    uint32_t nexts = 0U;
    dev->inst.vkEnumerateDeviceExtensionProperties(dev->phys, nullptr, &nexts, nullptr);
    VkExtensionProperties *exts =
        (VkExtensionProperties *)calloc(nexts ? nexts : 1U, sizeof(*exts));
    if (!exts) {
        return VELRI_ERR_NOMEM;
    }
    if (nexts > 0U) {
        dev->inst.vkEnumerateDeviceExtensionProperties(dev->phys, nullptr, &nexts, exts);
    }
    bool ext_coopmat = has_ext(exts, nexts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#ifdef VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME
    bool ext_cm2 = has_ext(exts, nexts, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV f_cm2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
    };
#endif
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    bool ext_bf16 = has_ext(exts, nexts, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    VkPhysicalDeviceShaderBfloat16FeaturesKHR f_bf16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
    };
#else
    bool ext_bf16 = false;
#endif
    free(exts);

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR f_cm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
    };
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &f12,
    };
    void **tail_pnext = &f13.pNext;
#ifdef VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME
    if (ext_cm2) {
        *tail_pnext = &f_cm2;
        tail_pnext = &f_cm2.pNext;
    }
#endif
    if (ext_coopmat) {
        *tail_pnext = &f_cm;
        tail_pnext = &f_cm.pNext;
    }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    if (ext_bf16) {
        *tail_pnext = &f_bf16;
    }
#else
    (void)ext_bf16;
    (void)tail_pnext;
#endif
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f11,
    };
    dev->inst.vkGetPhysicalDeviceFeatures2(dev->phys, &f2);

    out_feats->shader_int16 = f2.features.shaderInt16;
    out_feats->storage_16bit = f11.storageBuffer16BitAccess;
    out_feats->storage_8bit = f12.storageBuffer8BitAccess;
    out_feats->shader_float16 = f12.shaderFloat16;
    out_feats->shader_int8 = f12.shaderInt8;

    if (f2.features.shaderInt16 && f11.storageBuffer16BitAccess) {
        dev->caps_mask |= VK_CAP_INT16;
    }
    if ((dev->caps_mask & VK_CAP_INT16) && f12.shaderFloat16) {
        dev->caps_mask |= VK_CAP_F16_KV;
    }
    if ((dev->caps_mask & VK_CAP_INT16) && f12.shaderFloat16 && f12.shaderInt8 &&
        f12.storageBuffer8BitAccess) {
        dev->caps_mask |= VK_CAP_F16_I8;
    }
    if (f13.shaderIntegerDotProduct) {
        dev->caps_mask |= VK_CAP_INT_DOT;
    }
    if (max_smem >= 32768U) dev->caps_mask |= VK_CAP_SHARED_32K;
    if (max_smem >= VK_GEMM_F32_SMEM_BYTES) {
        dev->caps_mask |= VK_CAP_F32_GEMM;
    }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    if (ext_bf16 && f_bf16.shaderBFloat16Type) {
        dev->caps_mask |= VK_CAP_BF16_EXT;
    }
#endif

    bool has_coopmat = (ext_coopmat && f_cm.cooperativeMatrix == VK_TRUE);
    if (has_coopmat && dev->subgroup_size == 32U && max_smem >= VK_COOPMAT_128_SMEM_BYTES &&
        (dev->caps_mask & VK_CAP_F16_I8) &&
        dev->inst.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        uint32_t ncm = 0U;
        dev->inst.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev->phys, &ncm, nullptr);
        VkCooperativeMatrixPropertiesKHR *cm_props =
            (VkCooperativeMatrixPropertiesKHR *)calloc(ncm ? ncm : 1U, sizeof(*cm_props));
        if (cm_props && ncm > 0U) {
            for (uint32_t i = 0U; i < ncm; ++i) {
                cm_props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }
            dev->inst.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev->phys, &ncm, cm_props);
            for (uint32_t i = 0U; i < ncm; ++i) {
                const VkCooperativeMatrixPropertiesKHR *p = &cm_props[i];
                if (p->scope != VK_SCOPE_SUBGROUP_KHR || p->saturatingAccumulation) continue;
                if (p->MSize == 16U && p->NSize == 16U && p->KSize == 16U &&
                    p->AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    p->BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    p->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                    dev->caps_mask |= VK_CAP_CM_F16;
                }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
                if (p->MSize == 16U && p->NSize == 16U && p->KSize == 16U &&
                    p->AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    p->BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    p->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && ext_bf16 &&
                    f_bf16.shaderBFloat16CooperativeMatrix == VK_TRUE) {
                    dev->caps_mask |= VK_CAP_CM_BF16;
                }
#endif
                if (p->MSize == 16U && p->NSize == 16U && p->KSize == 32U &&
                    p->AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    p->BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    p->CType == VK_COMPONENT_TYPE_SINT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_SINT32_KHR) {
                    dev->caps_mask |= VK_CAP_CM_I8;
                }
            }
        }
        free(cm_props);
    }
#ifdef VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME
    if ((dev->caps_mask & VK_CAP_CM_F16) && f_cm2.cooperativeMatrixReductions &&
        f_cm2.cooperativeMatrixConversions && f_cm2.cooperativeMatrixPerElementOperations &&
        max_smem >= 49152U && limits->maxComputeWorkGroupSize[0] >= 256U) {
        dev->caps_mask |= VK_CAP_NV_ATTENTION;
    }
#endif
    return VELRI_OK;
}

static velri_status create_logical_device(velri_backend_device *dev,
                                          const vk_probed_feats *feats) {
    bool enable_cm = (dev->caps_mask & (VK_CAP_CM_F16 | VK_CAP_CM_BF16 | VK_CAP_CM_I8)) != 0U;
    const char *en_exts[4];
    uint32_t n_en_exts = 0U;
    if (enable_cm) {
        en_exts[n_en_exts++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    bool enable_bf16 = (dev->caps_mask & (VK_CAP_BF16_EXT | VK_CAP_CM_BF16)) != 0U;
    if (enable_bf16) {
        en_exts[n_en_exts++] = VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME;
    }
    VkPhysicalDeviceShaderBfloat16FeaturesKHR en_bf16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
        .shaderBFloat16Type = (dev->caps_mask & VK_CAP_BF16_EXT) ? VK_TRUE : VK_FALSE,
        .shaderBFloat16CooperativeMatrix = (dev->caps_mask & VK_CAP_CM_BF16) ? VK_TRUE : VK_FALSE,
    };
    void *bf16_next = enable_bf16 ? (void *)&en_bf16 : nullptr;
#else
    void *bf16_next = nullptr;
#endif
#ifdef VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV en_cm2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
        .pNext = bf16_next,
        .cooperativeMatrixReductions = VK_TRUE,
        .cooperativeMatrixConversions = VK_TRUE,
        .cooperativeMatrixPerElementOperations = VK_TRUE,
    };
    if (dev->caps_mask & VK_CAP_NV_ATTENTION) {
        en_exts[n_en_exts++] = VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME;
        bf16_next = &en_cm2;
    }
#endif
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_cm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = bf16_next,
        .cooperativeMatrix = enable_cm ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan13Features en13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = enable_cm ? (void *)&en_cm : bf16_next,
        .shaderIntegerDotProduct = (dev->caps_mask & VK_CAP_INT_DOT) ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan12Features en12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &en13,
        .storageBuffer8BitAccess = feats->storage_8bit,
        .shaderFloat16 = feats->shader_float16,
        .shaderInt8 = feats->shader_int8,
        .scalarBlockLayout = VK_TRUE,
        .vulkanMemoryModel = VK_TRUE,
        .vulkanMemoryModelDeviceScope = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceVulkan11Features en11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &en12,
        .storageBuffer16BitAccess = feats->storage_16bit,
    };
    VkPhysicalDeviceFeatures2 en2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &en11,
        .features = {
            .shaderInt16 = feats->shader_int16,
            .shaderInt64 = VK_TRUE,
        },
    };

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo q_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = dev->queue_family,
        .queueCount = 1U,
        .pQueuePriorities = &qprio,
    };
    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &en2,
        .queueCreateInfoCount = 1U,
        .pQueueCreateInfos = &q_ci,
        .enabledExtensionCount = n_en_exts,
        .ppEnabledExtensionNames = en_exts,
    };
    if (dev->inst.vkCreateDevice(dev->phys, &dev_ci, nullptr, &dev->device) != VK_SUCCESS) {
        return VELRI_ERR_UNSUPPORTED;
    }

    bool dev_ok = true;
#define LOAD_REQ_DEV(NAME)                                                          \
    do {                                                                            \
        PFN_vkVoidFunction _p = dev->inst.vkGetDeviceProcAddr(dev->device, #NAME);  \
        memcpy(&dev->NAME, &_p, sizeof(dev->NAME));                                 \
        if (!dev->NAME) dev_ok = false;                                             \
    } while (0);
    VK_DEV_FUNCS(LOAD_REQ_DEV)
#undef LOAD_REQ_DEV
    if (!dev_ok) {
        return VELRI_ERR_UNSUPPORTED;
    }

    dev->vkGetDeviceQueue(dev->device, dev->queue_family, 0U, &dev->queue);
    return VELRI_OK;
}

velri_status velri_vk_init_cmd_slot(velri_backend_device *dev, VkCommandPool *out_pool,
                                    VkCommandBuffer *out_cmd, VkFence *out_fence) {
    *out_pool = VK_NULL_HANDLE;
    *out_cmd = VK_NULL_HANDLE;
    *out_fence = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cp_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = dev->queue_family,
    };
    if (dev->vkCreateCommandPool(dev->device, &cp_ci, nullptr, out_pool) != VK_SUCCESS) {
        return VELRI_ERR_NOMEM;
    }
    VkCommandBufferAllocateInfo cb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = *out_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1U,
    };
    VkFenceCreateInfo fn_ci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (dev->vkAllocateCommandBuffers(dev->device, &cb_ai, out_cmd) != VK_SUCCESS ||
        dev->vkCreateFence(dev->device, &fn_ci, nullptr, out_fence) != VK_SUCCESS) {
        if (*out_fence) {
            dev->vkDestroyFence(dev->device, *out_fence, nullptr);
            *out_fence = VK_NULL_HANDLE;
        }
        dev->vkDestroyCommandPool(dev->device, *out_pool, nullptr);
        *out_pool = VK_NULL_HANDLE;
        *out_cmd = VK_NULL_HANDLE;
        return VELRI_ERR_NOMEM;
    }
    return VELRI_OK;
}

/* ---- 2. Device Lifecycle ---- */

static void release_logical_device(velri_backend_device *dev) {
    if (dev->device) {
        if (dev->queue && dev->vkQueueWaitIdle) {
            dev->vkQueueWaitIdle(dev->queue);
        }
        velri_vk_free_buf_locked(dev, dev->splitk_buf);
        dev->splitk_buf = nullptr;
        velri_vk_free_all_step_slabs_locked(dev);
        if (dev->stage_map && dev->vkUnmapMemory) {
            dev->vkUnmapMemory(dev->device, dev->stage_mem);
        }
        if (dev->stage_buf && dev->vkDestroyBuffer) {
            dev->vkDestroyBuffer(dev->device, dev->stage_buf, nullptr);
        }
        if (dev->stage_mem && dev->vkFreeMemory) {
            dev->vkFreeMemory(dev->device, dev->stage_mem, nullptr);
        }
        for (size_t i = 0U; i < (size_t)PIPE_COUNT; ++i) {
            if (dev->pipes[i] && dev->vkDestroyPipeline) {
                dev->vkDestroyPipeline(dev->device, dev->pipes[i], nullptr);
            }
        }
        if (dev->pipe_layout && dev->vkDestroyPipelineLayout) {
            dev->vkDestroyPipelineLayout(dev->device, dev->pipe_layout, nullptr);
        }
        for (uint32_t i = 0U; i < VK_CMD_RING_SIZE; ++i) {
            if (dev->ring[i].fence && dev->vkDestroyFence) {
                dev->vkDestroyFence(dev->device, dev->ring[i].fence, nullptr);
            }
            if (dev->ring[i].pool && dev->vkDestroyCommandPool) {
                dev->vkDestroyCommandPool(dev->device, dev->ring[i].pool, nullptr);
            }
        }
        if (dev->xfer_fence && dev->vkDestroyFence) {
            dev->vkDestroyFence(dev->device, dev->xfer_fence, nullptr);
        }
        if (dev->xfer_pool && dev->vkDestroyCommandPool) {
            dev->vkDestroyCommandPool(dev->device, dev->xfer_pool, nullptr);
        }
        if (dev->vkDestroyDevice) {
            dev->vkDestroyDevice(dev->device, nullptr);
        }
    }
    free(dev->enc_scratch);
}

void velri_vk_close_device(velri_backend_device *dev) {
    if (!dev) {
        return;
    }
    release_logical_device(dev);
    velri_os_cond_destroy(dev->slot_cv);
    velri_os_mutex_destroy(dev->mtx);
    vk_inst_destroy(&dev->inst);
    free(dev);
}

static velri_status init_logical_resources(velri_backend_device *dev) {
    velri_status st = create_logical_device(dev, &dev->features);
    if (st != VELRI_OK) {
        return st;
    }

    st = velri_vk_init_cmd_slot(dev, &dev->xfer_pool, &dev->xfer_cmd, &dev->xfer_fence);
    if (st != VELRI_OK) {
        return st;
    }
    for (uint32_t i = 0U; i < VK_CMD_RING_SIZE; ++i) {
        st = velri_vk_init_cmd_slot(dev, &dev->ring[i].pool, &dev->ring[i].cmd,
                                    &dev->ring[i].fence);
        if (st != VELRI_OK) {
            return st;
        }
    }

    dev->enc_scratch =
        (vk_encoded_dispatch *)calloc(VK_ENC_SCRATCH_INIT_CAP, sizeof(*dev->enc_scratch));
    if (!dev->enc_scratch) {
        return VELRI_ERR_NOMEM;
    }
    dev->enc_scratch_cap = VK_ENC_SCRATCH_INIT_CAP;

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0U,
        .size = VK_MAX_PC_BYTES,
    };
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &pcr,
    };
    if (dev->vkCreatePipelineLayout(dev->device, &pl_ci, nullptr, &dev->pipe_layout) !=
        VK_SUCCESS) {
        return VELRI_ERR_NOMEM;
    }

    return velri_vk_init_staging_buffer(dev);
}

/* Called before publishing the device or under its mutex. A failed activation releases
 * everything it acquired, so a later prepare or allocation may retry. */
velri_status velri_vk_activate_locked(velri_backend_device *dev) {
    if (dev->device) {
        return VELRI_OK;
    }
    velri_status st = init_logical_resources(dev);
    if (st != VELRI_OK) {
        release_logical_device(dev);
        dev->device = VK_NULL_HANDLE;
        dev->queue = VK_NULL_HANDLE;
        dev->pipe_layout = VK_NULL_HANDLE;
        dev->xfer_pool = VK_NULL_HANDLE;
        dev->xfer_cmd = VK_NULL_HANDLE;
        dev->xfer_fence = VK_NULL_HANDLE;
        memset(dev->ring, 0, sizeof(dev->ring));
        dev->enc_scratch = nullptr;
        dev->enc_scratch_cap = 0U;
    }
    return st;
}

velri_status velri_vk_open_device(uint32_t local_index, const void *cfg,
                                  velri_backend_device **out_dev) {
    if (!out_dev) {
        return VELRI_ERR_INVALID;
    }
    *out_dev = nullptr;

    velri_backend_device *dev = (velri_backend_device *)calloc(1U, sizeof(*dev));
    if (!dev) {
        return VELRI_ERR_NOMEM;
    }
    velri_status st = vk_inst_init(&dev->inst);
    if (st != VELRI_OK) {
        free(dev);
        return st;
    }

    vk_eligible_phys *list = nullptr;
    uint32_t count = 0U;
    st = collect_eligible_devices(&dev->inst, &list, &count);
    if (st != VELRI_OK || local_index >= count) {
        free(list);
        velri_vk_close_device(dev);
        return (st != VELRI_OK) ? st : VELRI_ERR_NOT_FOUND;
    }
    dev->phys = list[local_index].phys;
    dev->queue_family = list[local_index].queue_family;
    dev->local_index = local_index;
    free(list);

    VkPhysicalDeviceSubgroupProperties sg_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &sg_props,
    };
    dev->inst.vkGetPhysicalDeviceProperties2(dev->phys, &props2);
    dev->inst.vkGetPhysicalDeviceMemoryProperties(dev->phys, &dev->mem_props);
    populate_device_caps(dev, &props2, &sg_props);

    st = probe_features_and_coopmat(dev, &props2.properties.limits, &dev->features);
    if (st != VELRI_OK) {
        velri_vk_close_device(dev);
        return st;
    }

    if (!velri_os_mutex_create(&dev->mtx) || !velri_os_cond_create(&dev->slot_cv)) {
        velri_vk_close_device(dev);
        return VELRI_ERR_NOMEM;
    }

    /* Keep the primary Vulkan device eager. Additional enumerated devices acquire their
     * queues, pipelines and transfer resources when a plan actually selects them. */
    const velri_vk_open_cfg *vk_cfg = cfg;
    if (local_index == 0U || !vk_cfg || !vk_cfg->defer_secondary_devices) {
        st = velri_vk_activate_locked(dev);
        if (st != VELRI_OK) {
            velri_vk_close_device(dev);
            return st;
        }
    }
    *out_dev = dev;
    return VELRI_OK;
}

velri_status velri_vk_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps) {
    if (!dev || !out_caps) {
        return VELRI_ERR_INVALID;
    }
    *out_caps = dev->caps;
    return VELRI_OK;
}
