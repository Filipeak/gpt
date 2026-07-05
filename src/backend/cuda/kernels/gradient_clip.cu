#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

#define EPSILON 1e-6f

__global__ void clip_grad_norm_kernel(
    float *__restrict__ g,
    int size,
    const float *norm,
    float max_norm)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
    {
        return;
    }

    const float current_norm = *norm;
    if (current_norm <= max_norm)
    {
        return;
    }

    const float scale = max_norm / (current_norm + EPSILON);
    float4* g4 = (float4*)g + idx;

    g4->x *= scale;
    g4->y *= scale;
    g4->z *= scale;
    g4->w *= scale;
}

void CUDABackend::device_clip_grad_norm(float *g, int size, float max_norm)
{
    cublasPointerMode_t previous_pointer_mode;
    CUBLAS_CHECK(cublasGetPointerMode(cublas_handle_, &previous_pointer_mode));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CHECK(cublasSnrm2(cublas_handle_, size, g, 1, clip_norm_device_));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, previous_pointer_mode));

    size /= 4; // For vectorized float4 access

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    clip_grad_norm_kernel<<<grid_size, block_size>>>(
        g,
        size,
        clip_norm_device_,
        max_norm);

    CUDA_KERNEL_CHECK();
}