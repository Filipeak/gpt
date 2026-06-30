#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"

void CUDABackend::device_get_memory_info(float *free_mb, float *total_mb)
{
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    *free_mb = free_mem / (1024.0 * 1024.0);
    *total_mb = total_mem / (1024.0 * 1024.0);
}

void CUDABackend::device_malloc(void **ptr, size_t size)
{
    CUDA_CHECK(cudaMalloc(ptr, size));
}

void CUDABackend::device_free(void *ptr)
{
    CUDA_CHECK(cudaFree(ptr));
}

void CUDABackend::device_memset(void *ptr, int value, size_t size)
{
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

void CUDABackend::device_fill_normal(float *ptr, float mean, float stddev, size_t size)
{
    CURAND_CHECK(curandGenerateNormal(curand_handle_, ptr, size, mean, stddev));
}

__global__ void fill_const_kernel(float *ptr, float value, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        ptr[idx] = value;
    }
}

void CUDABackend::device_fill_const(float *ptr, float value, size_t size)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    fill_const_kernel<<<grid_size, block_size>>>(
        ptr,
        value,
        size);
}

void CUDABackend::device_memcpy_d2h(void *dst, const void *src, size_t size)
{
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void CUDABackend::device_memcpy_h2d(void *dst, const void *src, size_t size)
{
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}