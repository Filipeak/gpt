#pragma once

#include <chrono>

#define BENCHMARK_NOW() std::chrono::high_resolution_clock::now()
#define BENCHMARK_ELAPSED_MS(start, end) std::chrono::duration<double, std::milli>(end - start).count()

#define BENCHMARK_SCOPE(name, scope)     \
    auto start_##name = BENCHMARK_NOW(); \
    scope;                               \
    auto end_##name = BENCHMARK_NOW();   \
    float elapsed_##name = BENCHMARK_ELAPSED_MS(start_##name, end_##name);

#define BENCHMARK_SCOPE_PRINT(name, scope) \
    BENCHMARK_SCOPE(name, scope)           \
    LOG_INFO(#name " took %.2f ms", elapsed_##name);