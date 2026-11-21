#define VK_NO_PROTOTYPES

#include "os.h"
#include "reference.h"
#include "velri_lib/velri_lib.h"
#include "velri_vk_ceilings.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

/*
 * Vulkan Hardware Ceiling & Batched Kernel Roofline Benchmark.
 * Uses `velri_vk_ceilings.h` (benchmark-only SPIR-V shaders) for physical ceiling calibration
 * and runs production kernels through the public `VelriLib` batched `velri_slot` API.
 */

static void check_ok(velri_status st, const char *ctx) {
    if (st != VELRI_OK) {
        fprintf(stderr, "ERROR (%s): velri_status=%d\n", ctx, (int)st);
        exit(1);
    }
}

static void *xalloc(size_t bytes) {
    size_t sz = bytes ? ((bytes + 63U) & ~(size_t)63U) : 64U;
    void *p = velri_os_aligned_alloc(64U, sz);
    if (!p) {
        fprintf(stderr, "FATAL: aligned alloc failed\n");
        exit(1);
    }
    memset(p, 0, sz);
    return p;
}

typedef struct {
    uint64_t src, dst;
    uint32_t count_vec4, pad;
} ceil_bw_args;

typedef struct {
    uint64_t sink;
    uint32_t iters;
    float seed;
} ceil_comp_args;

typedef struct {
    double bw_copy_gbps;
    double bw_read_gbps;
    double fma_f32_tflops;
    double cm_f16_tflops;
    double cm_bf16_tflops;
    double cm_i8_tops;
} vk_ceilings;

/* Minimal standalone Vulkan timestamp runner for hardware ceiling calibration only. */
static vk_ceilings measure_vk_ceilings(void) {
    vk_ceilings out = {0};
    velri_os_lib *dl = velri_os_dlopen_vulkan();
    if (!dl) return out;

    PFN_vkGetInstanceProcAddr gipa = nullptr;
    void *sym = velri_os_dlsym(dl, "vkGetInstanceProcAddr");
    if (!sym) { velri_os_dlclose(dl); return out; }
    memcpy(&gipa, &sym, sizeof(gipa));

    PFN_vkCreateInstance pCreateInst = nullptr;
    PFN_vkVoidFunction fn = gipa(VK_NULL_HANDLE, "vkCreateInstance");
    memcpy(&pCreateInst, &fn, sizeof(pCreateInst));
    if (!pCreateInst) { velri_os_dlclose(dl); return out; }

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "velri-bench-vk",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst = VK_NULL_HANDLE;
    if (pCreateInst(&ici, nullptr, &inst) != VK_SUCCESS) { velri_os_dlclose(dl); return out; }

#define LD_I(N) PFN_##N N = nullptr; do { PFN_vkVoidFunction _f = gipa(inst, #N); memcpy(&N, &_f, sizeof(N)); } while (0)
    LD_I(vkDestroyInstance);
    LD_I(vkEnumeratePhysicalDevices);
    LD_I(vkGetPhysicalDeviceProperties2);
    LD_I(vkGetPhysicalDeviceFeatures2);
    LD_I(vkGetPhysicalDeviceMemoryProperties);
    LD_I(vkGetPhysicalDeviceQueueFamilyProperties);
    LD_I(vkCreateDevice);
    LD_I(vkGetDeviceProcAddr);
#undef LD_I

    uint32_t nd = 0U;
    vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    if (nd == 0U) { vkDestroyInstance(inst, nullptr); velri_os_dlclose(dl); return out; }
    VkPhysicalDevice *pds = (VkPhysicalDevice *)calloc(nd, sizeof(*pds));
    vkEnumeratePhysicalDevices(inst, &nd, pds);

    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t qf = UINT32_MAX;
    float ts_period = 0.0f;
    for (uint32_t i = 0U; i < nd; ++i) {
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        vkGetPhysicalDeviceProperties2(pds[i], &p2);
        if (p2.properties.apiVersion < VK_API_VERSION_1_3) continue;
        uint32_t nq = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &nq, nullptr);
        VkQueueFamilyProperties *qps = (VkQueueFamilyProperties *)calloc(nq ? nq : 1U, sizeof(*qps));
        vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &nq, qps);
        for (uint32_t q = 0U; q < nq; ++q) {
            if ((qps[q].queueFlags & VK_QUEUE_COMPUTE_BIT) && qps[q].timestampValidBits > 0U) {
                phys = pds[i];
                qf = q;
                ts_period = p2.properties.limits.timestampPeriod;
                break;
            }
        }
        free(qps);
        if (phys && p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
    }
    free(pds);
    if (!phys || ts_period <= 0.0f) {
        vkDestroyInstance(inst, nullptr);
        velri_os_dlclose(dl);
        return out;
    }

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    VkPhysicalDeviceVulkan12Features en12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .scalarBlockLayout = VK_TRUE,
        .vulkanMemoryModel = VK_TRUE,
        .vulkanMemoryModelDeviceScope = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceFeatures2 en2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &en12};
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qf,
        .queueCount = 1U, .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &en2,
        .queueCreateInfoCount = 1U, .pQueueCreateInfos = &qci,
    };
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
        vkDestroyInstance(inst, nullptr);
        velri_os_dlclose(dl);
        return out;
    }

#define LD_D(N) PFN_##N N = nullptr; do { PFN_vkVoidFunction _f = vkGetDeviceProcAddr(dev, #N); memcpy(&N, &_f, sizeof(N)); } while (0)
    LD_D(vkDestroyDevice); LD_D(vkGetDeviceQueue); LD_D(vkCreateCommandPool); LD_D(vkDestroyCommandPool);
    LD_D(vkAllocateCommandBuffers); LD_D(vkResetCommandBuffer); LD_D(vkBeginCommandBuffer);
    LD_D(vkEndCommandBuffer); LD_D(vkQueueSubmit); LD_D(vkCreateFence); LD_D(vkDestroyFence);
    LD_D(vkWaitForFences); LD_D(vkResetFences); LD_D(vkCreateQueryPool); LD_D(vkDestroyQueryPool);
    LD_D(vkGetQueryPoolResults); LD_D(vkCreateBuffer); LD_D(vkDestroyBuffer);
    LD_D(vkGetBufferMemoryRequirements); LD_D(vkAllocateMemory); LD_D(vkFreeMemory);
    LD_D(vkBindBufferMemory); LD_D(vkGetBufferDeviceAddress); LD_D(vkCreateShaderModule);
    LD_D(vkDestroyShaderModule); LD_D(vkCreatePipelineLayout); LD_D(vkDestroyPipelineLayout);
    LD_D(vkCreateComputePipelines); LD_D(vkDestroyPipeline); LD_D(vkCmdBindPipeline);
    LD_D(vkCmdPushConstants); LD_D(vkCmdDispatch); LD_D(vkCmdPipelineBarrier);
    LD_D(vkCmdFillBuffer); LD_D(vkCmdResetQueryPool); LD_D(vkCmdWriteTimestamp);
#undef LD_D

    VkQueue q = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, qf, 0U, &q);
    VkCommandPool cp = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qf};
    vkCreateCommandPool(dev, &cpci, nullptr, &cp);
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1U};
    vkAllocateCommandBuffers(dev, &cbai, &cb);
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(dev, &fci, nullptr, &fence);
    VkQueryPoolCreateInfo qpci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2U};
    vkCreateQueryPool(dev, &qpci, nullptr, &qp);
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 128U};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1U, .pPushConstantRanges = &pcr};
    vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

    /* Measure Read-Only BW (256 MiB) + Copy BW (256 MiB) + F32 FMA Peak */
    size_t bw_bytes = 256U * 1024U * 1024U;
    VkBuffer bufs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory mems[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceAddress bdas[2] = {0, 0};
    for (int i = 0; i < 2; ++i) {
        VkBufferCreateInfo bci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bw_bytes,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (vkCreateBuffer(dev, &bci, nullptr, &bufs[i]) == VK_SUCCESS) {
            VkMemoryRequirements mr;
            vkGetBufferMemoryRequirements(dev, bufs[i], &mr);
            uint32_t mt = UINT32_MAX;
            for (uint32_t t = 0U; t < mp.memoryTypeCount; ++t) {
                if ((mr.memoryTypeBits & (1U << t)) && (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = t; break; }
            }
            VkMemoryAllocateFlagsInfo afi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
            VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &afi, .allocationSize = mr.size, .memoryTypeIndex = mt};
            if (mt != UINT32_MAX && vkAllocateMemory(dev, &mai, nullptr, &mems[i]) == VK_SUCCESS &&
                vkBindBufferMemory(dev, bufs[i], mems[i], 0U) == VK_SUCCESS) {
                VkBufferDeviceAddressInfo bai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = bufs[i]};
                bdas[i] = vkGetBufferDeviceAddress(dev, &bai);
            }
        }
    }

    if (bdas[0] && bdas[1]) {
        const uint32_t *codes[3] = {spv_vk_ceiling_bw_f32x4, spv_vk_ceiling_bw_read_f32x4, spv_vk_ceiling_fma_f32};
        const size_t sizes[3] = {spv_vk_ceiling_bw_f32x4_bytes, spv_vk_ceiling_bw_read_f32x4_bytes, spv_vk_ceiling_fma_f32_bytes};
        VkPipeline pipes[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
        for (int i = 0; i < 3; ++i) {
            VkShaderModule sm = VK_NULL_HANDLE;
            VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizes[i], .pCode = codes[i]};
            if (vkCreateShaderModule(dev, &smci, nullptr, &sm) == VK_SUCCESS) {
                VkComputePipelineCreateInfo pci = {
                    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
                    .layout = pl,
                };
                vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1U, &pci, nullptr, &pipes[i]);
                vkDestroyShaderModule(dev, sm, nullptr);
            }
        }

        VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VkMemoryBarrier mbar = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1U, .pCommandBuffers = &cb};

        /* Commit pages */
        vkResetCommandBuffer(cb, 0U);
        vkBeginCommandBuffer(cb, &bi);
        vkCmdFillBuffer(cb, bufs[0], 0U, bw_bytes, 0x3F800000U);
        vkCmdFillBuffer(cb, bufs[1], 0U, bw_bytes, 0U);
        vkEndCommandBuffer(cb);
        vkResetFences(dev, 1U, &fence);
        vkQueueSubmit(q, 1U, &si, fence);
        vkWaitForFences(dev, 1U, &fence, VK_TRUE, UINT64_MAX);

        for (int k = 0; k < 3; ++k) {
            if (!pipes[k]) continue;
            uint32_t iters = (k == 2) ? 5U : 20U;
            uint32_t gx = (k == 0) ? (uint32_t)((bw_bytes / 16U + 1023U) / 1024U) : (k == 1 ? 2048U : 4096U);
            ceil_bw_args ba = {.src = bdas[0], .dst = bdas[1], .count_vec4 = (uint32_t)(bw_bytes / 16U)};
            ceil_comp_args ca = {.sink = bdas[1], .iters = 2000U, .seed = 0.5f};
            const void *pc = (k < 2) ? (const void *)&ba : (const void *)&ca;
            uint32_t pcsz = (k < 2) ? (uint32_t)sizeof(ba) : (uint32_t)sizeof(ca);

            vkResetCommandBuffer(cb, 0U);
            vkBeginCommandBuffer(cb, &bi);
            vkCmdResetQueryPool(cb, qp, 0U, 2U);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0U);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[k]);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0U, pcsz, pc);
            for (uint32_t it = 0U; it < iters; ++it) {
                vkCmdDispatch(cb, gx, 1U, 1U);
                if (it + 1U < iters) {
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &mbar, 0U, nullptr, 0U, nullptr);
                }
            }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1U);
            vkEndCommandBuffer(cb);
            vkResetFences(dev, 1U, &fence);
            vkQueueSubmit(q, 1U, &si, fence);
            vkWaitForFences(dev, 1U, &fence, VK_TRUE, UINT64_MAX);

            uint64_t ts[2] = {0, 0};
            if (vkGetQueryPoolResults(dev, qp, 0U, 2U, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS && ts[1] > ts[0]) {
                double sec = ((double)(ts[1] - ts[0]) * (double)ts_period * 1e-9) / (double)iters;
                if (k == 0) out.bw_copy_gbps = (2.0 * (double)bw_bytes) / sec / 1e9;
                else if (k == 1) out.bw_read_gbps = (double)bw_bytes / sec / 1e9;
                else out.fma_f32_tflops = (4096.0 * 256.0 * 2000.0 * 512.0) / sec / 1e12;
            }
            vkDestroyPipeline(dev, pipes[k], nullptr);
        }
    }

    for (int i = 0; i < 2; ++i) {
        if (bufs[i]) vkDestroyBuffer(dev, bufs[i], nullptr);
        if (mems[i]) vkFreeMemory(dev, mems[i], nullptr);
    }
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyQueryPool(dev, qp, nullptr);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cp, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    velri_os_dlclose(dl);
    return out;
}

/* Benchmark a Vulkan Linear op via batched `velri_slot` (resident WEIGHT + STATE, 0 PCIe traffic). */
static double bench_vk_linear_ms(velri_context *ctx, velri_device_id dev, velri_linear_desc d,
                                 size_t rows, size_t m, size_t k, uint32_t batch_reps) {
    size_t xb = velri_dtype_bytes(d.x_dtype);
    size_t wb = velri_dtype_bytes(d.w_dtype);
    size_t yb = velri_dtype_bytes(d.y_dtype);

    velri_graph *g = nullptr;
    check_ok(velri_graph_create(&g), "g");
    velri_storage_id sx = 0U, sw = 0U, sy = 0U, sxs = 0U, sws = 0U;
    velri_storage_desc dx = {VELRI_STORAGE_WEIGHT, d.x_dtype, rows * k * xb, 256U, dev};
    velri_storage_desc dw = {VELRI_STORAGE_WEIGHT, d.w_dtype, m * k * wb, 256U, dev};
    velri_storage_desc dy = {VELRI_STORAGE_STATE, d.y_dtype, rows * m * yb, 256U, dev};
    check_ok(velri_graph_add_storage(g, &dx, &sx), "sx");
    check_ok(velri_graph_add_storage(g, &dw, &sw), "sw");
    check_ok(velri_graph_add_storage(g, &dy, &sy), "sy");
    if (d.has_x_scale) {
        velri_storage_desc dxs = {VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, rows * sizeof(float), 256U, dev};
        check_ok(velri_graph_add_storage(g, &dxs, &sxs), "sxs");
    }
    if (d.has_w_scale) {
        velri_storage_desc dws = {VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * sizeof(float), 256U, dev};
        check_ok(velri_graph_add_storage(g, &dws, &sws), "sws");
    }

    size_t xd[2] = {rows, k}, wd[2] = {m, k}, yd[2] = {rows, m}, xsd[1] = {rows}, wsd[1] = {m};
    velri_tensor_view vxt, vwt, vyt, vxst, vwst;
    velri_view_id vx = 0U, vw = 0U, vy = 0U, vxs = VELRI_INVALID_ID, vws = VELRI_INVALID_ID;
    check_ok(velri_tensor_view_contiguous(sx, d.x_dtype, 2U, xd, 0U, &vxt), "vx");
    check_ok(velri_tensor_view_contiguous(sw, d.w_dtype, 2U, wd, 0U, &vwt), "vw");
    check_ok(velri_tensor_view_contiguous(sy, d.y_dtype, 2U, yd, 0U, &vyt), "vy");
    check_ok(velri_graph_add_view(g, &vxt, &vx), "avx");
    check_ok(velri_graph_add_view(g, &vwt, &vw), "avw");
    check_ok(velri_graph_add_view(g, &vyt, &vy), "avy");
    if (d.has_x_scale) {
        check_ok(velri_tensor_view_contiguous(sxs, VELRI_DTYPE_F32, 1U, xsd, 0U, &vxst), "vxs");
        check_ok(velri_graph_add_view(g, &vxst, &vxs), "avxs");
    }
    if (d.has_w_scale) {
        check_ok(velri_tensor_view_contiguous(sws, VELRI_DTYPE_F32, 1U, wsd, 0U, &vwst), "vws");
        check_ok(velri_graph_add_view(g, &vwst, &vws), "avws");
    }

    velri_op_desc op = {
        .kind = VELRI_OP_LINEAR, .num_inputs = 5U, .num_outputs = 1U,
        .inputs = {vx, vw, VELRI_INVALID_ID, vxs, vws}, .outputs = {vy},
        .attr.linear.desc = d,
    };
    for (uint32_t i = 0U; i < batch_reps; ++i) {
        check_ok(velri_graph_add_op(g, &op, dev, nullptr), "add op");
    }

    velri_plan *plan = nullptr;
    velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
    if (pst == VELRI_ERR_UNSUPPORTED) {
        velri_graph_destroy(g);
        return -1.0;
    }
    check_ok(pst, "prep");

    uint64_t rng = 0xD0000001ULL;
    void *hx = xalloc(rows * k * xb);
    void *hw = xalloc(m * k * wb);
    void *hy = xalloc(rows * m * yb);
    float *hxs = d.has_x_scale ? (float *)xalloc(rows * sizeof(float)) : nullptr;
    float *hws = d.has_w_scale ? (float *)xalloc(m * sizeof(float)) : nullptr;
    velri_test_fill_dtype(d.x_dtype, hx, rows * k, &rng, 0.5f);
    velri_test_fill_dtype(d.w_dtype, hw, m * k, &rng, 0.5f);
    if (hxs) velri_test_fill_dtype(VELRI_DTYPE_F32, hxs, rows, &rng, 0.02f);
    if (hws) velri_test_fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.02f);

    check_ok(velri_plan_bind_weight_host(plan, sx, hx, rows * k * xb), "bx");
    check_ok(velri_plan_bind_weight_host(plan, sw, hw, m * k * wb), "bw");
    if (hxs) check_ok(velri_plan_bind_weight_host(plan, sxs, hxs, rows * sizeof(float)), "bxs");
    if (hws) check_ok(velri_plan_bind_weight_host(plan, sws, hws, m * sizeof(float)), "bws");

    velri_slot *slot = nullptr;
    check_ok(velri_slot_create(plan, &slot), "slot");
    check_ok(velri_slot_bind_host(slot, sy, hy, rows * m * yb), "by");
    check_ok(velri_slot_run(slot, nullptr, nullptr), "warmup");

    uint64_t t0 = velri_os_monotonic_ns();
    for (int s = 0; s < 3; ++s) {
        check_ok(velri_slot_run(slot, nullptr, nullptr), "run");
    }
    double avg_ms = ((double)(velri_os_monotonic_ns() - t0) * 1e-6) / (3.0 * (double)batch_reps);

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    velri_os_aligned_free(hx); velri_os_aligned_free(hw); velri_os_aligned_free(hy);
    velri_os_aligned_free(hxs); velri_os_aligned_free(hws);
    return avg_ms;
}

int main(void) {
    velri_context_config cfg = {.cpu_workers = 1U, .cpu_variant = VELRI_CPU_AUTO, .enable_vulkan = true};
    velri_context *ctx = nullptr;
    check_ok(velri_context_create(&cfg, &ctx), "ctx");

    velri_device_id vk_dev = 0U;
    if (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) != VELRI_OK) {
        printf("No Vulkan 1.3 GPU found.\n");
        velri_context_destroy(ctx);
        return 0;
    }

    velri_device_caps caps;
    check_ok(velri_context_device_caps(ctx, vk_dev, &caps), "caps");
    printf("Velri-lib Vulkan Roofline Benchmark — %s (UMA=%d, VRAM=%.1f GiB)\n",
           caps.device_name, (int)caps.is_uma, (double)caps.mem_total_bytes / (1024.0 * 1024.0 * 1024.0));

    vk_ceilings ceil = measure_vk_ceilings();
    printf("\n=== Measured GPU Physical Ceilings ===\n");
    printf("  VRAM Read-Only BW (256 MiB) : %7.1f GB/s\n", ceil.bw_read_gbps);
    printf("  VRAM Copy BW (256->256 MiB) : %7.1f GB/s\n", ceil.bw_copy_gbps);
    printf("  FP32 FMA Peak               : %7.2f TFLOPS\n", ceil.fma_f32_tflops);

    struct {
        const char *name;
        velri_linear_desc desc;
    } modes[] = {
        {"F32", velri_desc_f32()},
        {"F16->F32", velri_desc_f16(VELRI_DTYPE_F32)},
        {"BF16->F32", velri_desc_bf16(VELRI_DTYPE_F32)},
        {"W8A8->F32", velri_desc_w8a8_f32()},
    };

    printf("\n=== Vulkan GEMV (rows=1, M=14336, K=4096) ===\n");
    printf("%-11s | %8s | %11s | %10s\n", "Mode", "Time ms", "Eff GB/s", "%Meas Read");
    printf("------------+----------+-------------+-----------\n");
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        velri_linear_desc d = modes[i].desc;
        double ms = bench_vk_linear_ms(ctx, vk_dev, d, 1U, 14336U, 4096U, 16U);
        if (ms <= 0.0) continue;
        double bytes = (double)(14336U * 4096U * velri_dtype_bytes(d.w_dtype) +
                                4096U * velri_dtype_bytes(d.x_dtype) +
                                14336U * velri_dtype_bytes(d.y_dtype));
        double gbps = bytes / (ms * 1e-3) / 1e9;
        printf("%-11s | %8.4f | %6.1f GB/s | %9.1f%%\n", modes[i].name, ms, gbps,
               ceil.bw_read_gbps > 0.0 ? 100.0 * gbps / ceil.bw_read_gbps : 0.0);
    }

    printf("\n=== Vulkan GEMM (rows=2048, M=2048, K=2048) ===\n");
    printf("%-11s | %8s | %12s\n", "Mode", "Time ms", "Throughput");
    printf("------------+----------+-------------\n");
    double ops = 2.0 * 2048.0 * 2048.0 * 2048.0;
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        velri_linear_desc d = modes[i].desc;
        double ms = bench_vk_linear_ms(ctx, vk_dev, d, 2048U, 2048U, 2048U, 8U);
        if (ms <= 0.0) continue;
        double tops = ops / (ms * 1e-3) / 1e12;
        printf("%-11s | %8.3f | %6.2f %s\n", modes[i].name, ms, tops,
               d.x_dtype == VELRI_DTYPE_I8 ? "TOPS" : "TFLOPS");
    }

    velri_context_destroy(ctx);
    return 0;
}
