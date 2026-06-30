#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#define CUDA_CHECK(expr_to_check)                         \
    do                                                    \
    {                                                     \
        cudaError_t result = expr_to_check;               \
        if (result != cudaSuccess)                        \
        {                                                 \
            printf("CUDA Runtime Error: %s:%i:%d = %s\n", \
                   __FILE__,                              \
                   __LINE__,                              \
                   result,                                \
                   cudaGetErrorString(result));           \
                                                          \
            abort();                                      \
        }                                                 \
    } while (0)

#define CUBLAS_CHECK(expr_to_check)                    \
    do                                                 \
    {                                                  \
        cublasStatus_t result = expr_to_check;         \
        if (result != CUBLAS_STATUS_SUCCESS)           \
        {                                              \
            printf("cuBLAS Runtime Error: %s:%i:%d\n", \
                   __FILE__,                           \
                   __LINE__,                           \
                   result);                            \
                                                       \
            abort();                                   \
        }                                              \
    } while (0)

#define CURAND_CHECK(expr_to_check)                    \
    do                                                 \
    {                                                  \
        curandStatus_t result = expr_to_check;         \
        if (result != CURAND_STATUS_SUCCESS)           \
        {                                              \
            printf("cuRAND Runtime Error: %s:%i:%d\n", \
                   __FILE__,                           \
                   __LINE__,                           \
                   result);                            \
                                                       \
            abort();                                   \
        }                                              \
    } while (0)

#ifndef NDEBUG
#define CUDA_DEBUG_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#else
#define CUDA_DEBUG_SYNC() ((void)0)
#endif