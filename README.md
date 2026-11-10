# Velri-lib

Velri-lib is the C23 computation library for [Velri](https://github.com/velri-inference/velri) and other clients. It owns
mathematical operations, tensor and storage descriptions, computation graphs,
execution planning, and device backends. CPU and Vulkan are the first target
backends. The library does not load model-family configurations, tokenize text,
manage chat requests, or serve HTTP.

Documentation: **[English](docs/EN/architecture.md)** · **[Русский](docs/RU/architecture.md)**.

## What works today

The extracted CPU linear module provides prepared GEMV/GEMM, baseline/AVX2/AVX-512
variants, an independent reference and a frozen historical fixture. It currently
accepts FP32 data and accumulates in ordered FP64. This is a retained experimental
component, **not implementation of the target F64, FP16, BF16 and INT8 family**.
Its arithmetic and exactness tests have not been relaxed during extraction.

Graphs, a common executor, additional data types and Vulkan remain planned work.
The `VelriLib::cpu_linear` CMake target exposes only this existing module. The
header is experimental and carries no stable ABI promise.

## Build and test

Use CMake 3.21+, Ninja, and Clang 18+ or GCC 14+ with C23 support:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

The standalone build enables tests. As a subproject, tests are opt-in through
`VELRI_LIB_BUILD_TESTS`. Existing CPU options remain `VELRI_CPU_MULTIVERSION`,
`VELRI_SANITIZE`, `VELRI_BUILD_BENCHMARK` and `VELRI_BUILD_PARALLEL_BENCHMARK`.
The last option builds a Linux diagnostic worker pool, not the future executor.

## Use from another CMake project

With a source checkout:

```cmake
add_subdirectory(path/to/Velri-lib velri-lib-build)
target_link_libraries(my_application PRIVATE VelriLib::cpu_linear)
```

Or install a package:

```sh
cmake --install build --prefix /path/to/velri-lib-install
```

```cmake
find_package(VelriLib 0.1.0 EXACT CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE VelriLib::cpu_linear)
```

Consumers include `<velri_lib/linear.h>`. No Rust, LLVM runtime, Vulkan SDK or
network download is required for the current CPU build. Source dependencies must
use a reviewed fixed revision when published; the engine must not track a moving
library branch automatically.

The numerical contract and historical measurements are in
[CPU linear](docs/EN/cpu-linear.md). They describe the retained prototype, not
performance promises for new data types. Raw benchmark reports stay out of Git.
