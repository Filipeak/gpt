#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

#define EPSILON 1e-6f

__global__ void scale_and_clip_grad_norm_f32_v4_kernel(
    float *__restrict__ g,
    int size,
    const float *norm,
    float max_norm,
    float grad_scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
    {
        return;
    }

    const float current_norm = *norm;
    const float scale = fminf(grad_scale, max_norm / (current_norm + EPSILON));

    float4 *g4 = (float4 *)g;
    float4 current = g4[idx];

    current.x *= scale;
    current.y *= scale;
    current.z *= scale;
    current.w *= scale;

    g4[idx] = current;
}

void CUDABackend::device_scale_and_clip_grad(float *g, int size, float max_norm, int accum_steps)
{
    cublasPointerMode_t previous_pointer_mode;
    CUBLAS_CHECK(cublasGetPointerMode(cublas_handle_, &previous_pointer_mode));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CHECK(cublasSnrm2(cublas_handle_, size, g, 1, clip_norm_device_));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, previous_pointer_mode));

    CUDA_ASSERT(size % 4 == 0); // Ensure size is a multiple of 4 for float4 processing
    size /= 4;                  // For vectorized float4 access

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    float grad_scale = 1.0f / accum_steps;

    scale_and_clip_grad_norm_f32_v4_kernel<<<grid_size, block_size>>>(
        g,
        size,
        clip_norm_device_,
        max_norm,
        grad_scale);

    CUDA_KERNEL_CHECK();
}