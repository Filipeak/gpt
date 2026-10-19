#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#define CUDA_ASSERT(expr)                                     \
    do                                                        \
    {                                                         \
        if (!(expr))                                          \
        {                                                     \
            fprintf(stderr, "CUDA Assertion Failed: %s:%i\n", \
                    __FILE__,                                 \
                    __LINE__);                                \
            abort();                                          \
        }                                                     \
    } while (0)

#define CUDA_CHECK(expr_to_check)                                  \
    do                                                             \
    {                                                              \
        cudaError_t result = expr_to_check;                        \
        if (result != cudaSuccess)                                 \
        {                                                          \
            fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", \
                    __FILE__,                                      \
                    __LINE__,                                      \
                    result,                                        \
                    cudaGetErrorString(result));                   \
            abort();                                               \
        }                                                          \
    } while (0)

#define CUBLAS_CHECK(expr_to_check)                             \
    do                                                          \
    {                                                           \
        cublasStatus_t result = expr_to_check;                  \
        if (result != CUBLAS_STATUS_SUCCESS)                    \
        {                                                       \
            fprintf(stderr, "cuBLAS Runtime Error: %s:%i:%d\n", \
                    __FILE__,                                   \
                    __LINE__,                                   \
                    result);                                    \
            abort();                                            \
        }                                                       \
    } while (0)

#define CURAND_CHECK(expr_to_check)                             \
    do                                                          \
    {                                                           \
        curandStatus_t result = expr_to_check;                  \
        if (result != CURAND_STATUS_SUCCESS)                    \
        {                                                       \
            fprintf(stderr, "cuRAND Runtime Error: %s:%i:%d\n", \
                    __FILE__,                                   \
                    __LINE__,                                   \
                    result);                                    \
            abort();                                            \
        }                                                       \
    } while (0)

// Catches launch errors only. Errors while a kernel runs show up at the next sync;
// build with -DGPT_DEBUG=ON to sync and check after every op (see CUDABackend::profile_range_pop).
#define CUDA_KERNEL_CHECK()             \
    do                                  \
    {                                   \
        CUDA_CHECK(cudaGetLastError()); \
    } while (0)