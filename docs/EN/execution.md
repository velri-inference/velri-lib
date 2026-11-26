# Preparation, Memory, and Hybrid Execution

[Русский](../RU/execution.md)

This document defines the execution model of `Velri-lib`: what runs during preparation, how CPU and Vulkan cooperate, how memory visibility is synchronized, and how errors and cancellation behave. For overall library boundaries see [architecture.md](architecture.md); for kernels see [kernels.md](kernels.md).

**Status:** the core execution model (`velri_plan_prepare`, scratch arena packing, post-packing DAG barrier elision, RAM transfer bus, cross-plan shared WEIGHT slabs via `share_weights_from`, cross-slot shared STATE arenas via `velri_slot_create_shared_state`, persistent Vulkan decode command capture/replay with bucketed FlashDecoding Split-KV, completion draining, and cancellation) is implemented and verified across CPU, `mock_async`, and optional Vulkan ([architecture.md](architecture.md)); timeline semaphores, `synchronization2`, and disk staging (PLE) follow in later stages.

---

## 1. Preparation (`prepare`) vs. Execution Step

Everything invariant stays out of the token-generation loop:

| Done once in `prepare` | Done on every step |
| :--- | :--- |
| Validate graph, shapes, dtypes, strides, and state accesses | Check dynamic bounds (token count, positions, MoE routes) |
| Query allowed devices (`query_operation`) | Populate compact dynamic parameters in an execution slot |
| Choose placement, AOT kernels, layouts, and explicit transfers/conversions | Submit prepared work to backends (`submit`) |
| Compute full memory budget and reserve buffers/slots | Wait for completion (`wait`) and return outputs |
| Pack immutable weights once and build GPU pipelines / descriptors | `0 B` heap allocations, no JSON parsing or string lookup |

The plan records the length of each same-device submission and its capture index. A transfer ends the batch. Slot creation and execution use the recorded boundaries, including single-operation submissions for backends without batch support.

If raw weights are no longer needed after packing, their buffer is released explicitly. Whenever both representations are alive simultaneously, both count toward the memory budget.

### Memory budgets and UMA
Budgets account for all concurrently live allocations: raw and packed weights, session state, `scratch`, staging buffers, alignment padding, and real allocator overhead (e.g., VMA). Driver-internal memory without a reliable query is reported as **unknown**.

On Unified Memory Architecture (**UMA**) systems (integrated GPUs, Android), CPU and GPU share physical RAM: host-visible and device-local pools must never be summed as two independent memory capacities.

---

## 2. Hybrid CPU + Vulkan Execution

The baseline transfer path between any two backends routes through host RAM:

```text
Device A resource  ──►  agreed RAM staging buffer  ──►  Device B resource
```

- Copying bytes does not convert internal layouts: when a layout is opaque to the consumer backend, `prepare` inserts an explicit supported conversion or rejects the placement.
- Staging slots cannot be reused until every consumer finishes reading them; slot reuse tracks concrete accepted work IDs rather than model layer numbers.

### Typical hybrid path (e.g., MoE experts on CPU, attention on Vulkan)
1. The Vulkan backend computes router selections and activations, copies them into a host-visible staging slot, and signals completion.
2. The CPU backend waits for completion and memory visibility, then runs selected experts across `velri_pool` using precomputed slices.
3. The CPU completes writes (issuing a `flush` for non-coherent memory when needed), transfers results back to GPU, and the dependent GPU `dispatch` waits on that transfer.
4. The staging slot returns to the pool only after its last consumer completes.

### Disk as a data source
Large disk-backed tables (e.g., n-gram PLE tables; see [qwen4_exp.md](https://github.com/velri-inference/velri/blob/main/docs/EN/models/qwen4_exp.md)) use bounded staging buffers with an explicit dependency between row loading/decoding and the first compute consumer. File I/O lives in the platform layer; disk benchmarks explicitly account for OS page cache effects.

---

## 3. Vulkan Synchronization and Memory

- **Baseline is Vulkan 1.1:** `VkFence` for host waits, binary `VkSemaphore` for cross-queue dependencies, and precise memory/execution barriers within a queue.
- **Opt-in extensions:** `timeline semaphores` (Vulkan 1.2 / `VK_KHR_timeline_semaphore`) and `synchronization2` are enabled only when confirmed supported.
- **No anti-patterns:** `vkQueueWaitIdle` per operation is forbidden as a general execution model; a single completion object tracks a batch of commands rather than each individual `dispatch`.
- **Mapped memory (`HOST_VISIBLE`):** `HOST_VISIBLE` alone guarantees neither fast CPU nor fast GPU access. Non-coherent memory requires proper `flush`/`invalidate` ranges aligned to `nonCoherentAtomSize`. Even on UMA or coherent memory, C atomics and `volatile` **do not replace** Vulkan synchronization and memory visibility rules.
- **Direct access and zero-copy:** skipping copies on UMA or direct discrete-GPU reads from `HOST_VISIBLE` memory are opt-in paths selected only from memory properties and measurements.
- Each active Vulkan device uses a transfer buffer of up to 4 MiB. Larger copies are submitted in chunks. The built-in backend lists secondary devices at context creation and opens their compute queues and transfer resources on first preparation or allocation.
- For slot I/O arenas up to 256 KiB, Vulkan tries to map coherent, device-local memory. The CPU can then write input tokens directly before submitting compute. Larger arenas and devices where mapping fails use staged uploads. Readbacks always use a coherent staging buffer, preferably one cached by the CPU.
- When the last Vulkan batch replays a captured graph, it can copy its outputs into a private CPU-cached buffer before signaling the same fence. The planner budgets the aligned span containing those outputs, up to 256 KiB per slot. Larger spans and devices without a suitable allocation use the regular staged copy.
- **Command buffer reuse:** command buffers are reused only with compatible resource bindings and safe slot lifetimes (Vulkan command reuse does not promise CUDA Graphs update semantics).

---

## 4. Completion, Errors, and Cancellation

The initial public engine step is **synchronous** to the caller, while backends inside the step may execute batches asynchronously through completion objects (`poll` / `wait`). Before returning, all accepted work and output transfers must finish.

### Resource safety rules
1. **Rejection before acceptance (`submit` fails without accepting work):** the device launched nothing and retains no caller buffers.
2. **Failure after partial acceptance:** the backend still returns a completion object for all launched memory accesses; the runtime waits until they finish before freeing or reusing buffers.
3. **In-flight cancellation:** the runtime stops submitting new work and waits for accepted work to complete. Forcibly aborting an in-flight kernel is not required from backends.
   - Cancellation before the first work of a step is accepted leaves prior session state intact.
   - Cancellation after work is accepted (or an error that may have touched state) marks the session as requiring an explicit `reset` before continuing; automatic partial-step rollback is not promised in v1.
   - Cancellation arriving after a step commits its successful result does not revoke the finished output.
4. **Device loss (`VK_ERROR_DEVICE_LOST`):** the backend runs its shutdown/cleanup path; affected resources are not reused until safe release is confirmed, and associated sessions transition to an error state.
