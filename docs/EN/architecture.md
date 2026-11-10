# What belongs in Velri-lib

[Русский](../RU/architecture.md)

Velri-lib executes described computations. Velri describes models and manages
inference requests. Both projects use C23; the dependency goes from Velri to
Velri-lib. The library must build and run tests without the engine.

## Responsibilities

| Velri-lib | Velri |
| :--- | :--- |
| Tensor views, storage, layouts and number types | Model configuration and weight-name mapping |
| Mathematical operations and reusable computation blocks | Selection and arrangement of blocks for a model family |
| Computation graph and validation | Construction of a model graph through the library API |
| Kernel selection, fusion, transfers and memory planning | Placement policy, resource budgets and workload limits |
| Prepared execution plan, job dependencies and completion | Requests, token history, batching policy and generation loop |
| CPU workers, Vulkan queues and backend resources | Tokenization, sampling policy, stopping rules and HTTP |
| Packing and mathematical decoding of declared weight formats | File/container reading, metadata and format adapters |
| Independent operation references and kernel tests | Model fixtures, logits and end-to-end quality checks |

Reusable MLP, attention, MoE and recurrent computations belong in the library
when described by explicit mathematical parameters. A model adapter chooses and
connects them. Neither a common block nor a backend branches on a model-family
name. Sampling can use library reductions, sorting or random-number primitives;
the engine owns the token-selection policy and generation state.

## From graph to execution

```text
Velri: model configuration + weights
                 |
                 v
Velri-lib: operations + values + declared state access
                 |
                 v
validation -> kernel/layout selection -> memory and transfer plan
                 |
                 v
prepared jobs -> CPU / Vulkan / another enabled backend
                 |
                 v
Velri: outputs + completion status -> next request step
```

The computation graph says what to calculate and which values or state regions
each operation reads or writes. The execution plan says how selected kernels,
copies, conversions and synchronization execute it on available devices. They
are a description and its prepared result, not two separate model compilers.
Model adapters do not specify AVX instructions, Vulkan barriers or worker IDs.

Preparation is outside the repeated execution path. It validates constant
geometry, selects implementations, packs immutable weights, reserves workspace
and prepares reusable jobs. Each invocation supplies bounded dynamic values,
input/output bindings and an execution slot. No graph rebuild or weight packing
is required merely because a token position or active sequence count changes
within the prepared bounds.

A backend queries support for an operation, prepares it and reports completion.
Backends can offer fused regions without being required to compile the whole
graph themselves. Missing support is resolved during preparation through an
allowed explicit decomposition or placement; a running kernel never changes
device secretly. Disabled backends add no mandatory build dependencies.

## Ownership and two kinds of scheduling

Velri owns model lifetime, request state and the decision about which sequences
share a batch. Velri-lib owns execution resources and schedules jobs within a
prepared invocation. Two projects must not create competing CPU worker pools.
The engine supplies budgets and concurrency limits to the library context.

A model can share immutable prepared weights and plans across requests. Each
active invocation needs separate mutable bindings, workspace and completion
records. Persistent state uses explicitly declared regions; K and V are roles,
not instructions to allocate two buffers. Aliases and reconstructed values stay
visible to the graph and memory planner.

Accepted work keeps its borrowed resources alive until completion, including
after cancellation or failure. The library drains device work and reports which
state may be invalid. Velri decides whether a request stops or requires reset.
There is no promise of rollback from a partially executed step.

## Data types and numerical accuracy

Target formats are **F64 (FP64), FP16, BF16 and INT8**. An operation separately
declares input/weight storage, accumulator type and output type. FP32 can be an
internal accumulator or conversion representation; it is not the sole model
format. INT8 weights with floating activations and INT8-by-INT8 computation have
different descriptors and quantization parameters. Support is queried for the
complete combination on the actual device.

Each floating operation has a documented error contract checked against an
independent higher-precision reference. Fast implementations may use FMA and
different reduction trees where the contract allows them. A blanket requirement
to reproduce one sequential FP64 loop is not imposed on future kernels. Exact
integer operations, storage transforms and index calculations remain exact.
Repeatability and agreement with a reference are separate tests. Tolerances must
account for dimensions, magnitudes and cancellation; changing a tolerance merely
to accept a broken implementation is not an optimization.

## Current implementation and next boundary check

Only the existing CPU linear module has been extracted. Its FP32 inputs and
ordered FP64 arithmetic remain unchanged for regression checking. The target
formats, graph, generic context/execution API and Vulkan are not implemented.
Do not describe the legacy FP32 prototype as an F64 or FP16/BF16/INT8 kernel.

The next complete path is a tiny standalone client that constructs operations,
prepares a plan, binds two independent execution slots and runs it through CPU.
It must work without a tokenizer, model loader or engine checkout. A limited
test backend must be attachable without editing model code or adding backend-name
branches in the planner. These tests establish the boundary before more backends
or a stable binary plugin ABI are added.

## Source layout

| Library directory | Purpose |
| :--- | :--- |
| `include/velri_lib/` | Client C headers; currently only experimental `linear.h` |
| `src/core/` | Planned values, storage descriptors and checked dimensions |
| `src/ops/`, `src/blocks/` | Planned mathematical operations and reusable compositions |
| `src/graph/`, `src/plan/` | Planned graph validation and execution preparation |
| `src/runtime/` | Planned context, execution slots, dependencies and completion |
| `src/backend/` | CPU module; planned common contract, Vulkan and future modules |
| `src/platform/` | Planned low-level threads, clocks and hardware discovery |
| `tests/`, `tools/`, `cmake/` | References, fixtures, measurements and standalone build |

Directories marked planned are not empty APIs pretending to be implementations.
The engine links through C headers and CMake targets; it must not include private
`src/` headers or compile copies of the library sources. Pin a library revision
when distributing the engine. Until the API settles, source compatibility and
version checks take precedence over promising a stable ABI.
