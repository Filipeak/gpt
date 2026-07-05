#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

__global__ void adamw_step_f32_v4_kernel(
    float *__restrict__ params,
    const float *__restrict__ g,
    float *__restrict__ m,
    float *__restrict__ v,
    int size,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps,
    float weight_decay)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float4 *params4 = (float4 *)params + idx;
        const float4 *g4 = (const float4 *)g + idx;
        float4 *m4 = (float4 *)m + idx;
        float4 *v4 = (float4 *)v + idx;

        float m_hat, v_hat;

        // First component
        m4->x = beta1 * m4->x + (1.0f - beta1) * g4->x;
        v4->x = beta2 * v4->x + (1.0f - beta2) * g4->x * g4->x;
        m_hat = m4->x * bias_correction1;
        v_hat = v4->x * bias_correction2;
        params4->x -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params4->x);

        // Second component
        m4->y = beta1 * m4->y + (1.0f - beta1) * g4->y;
        v4->y = beta2 * v4->y + (1.0f - beta2) * g4->y * g4->y;
        m_hat = m4->y * bias_correction1;
        v_hat = v4->y * bias_correction2;
        params4->y -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params4->y);

        // Third component
        m4->z = beta1 * m4->z + (1.0f - beta1) * g4->z;
        v4->z = beta2 * v4->z + (1.0f - beta2) * g4->z * g4->z;
        m_hat = m4->z * bias_correction1;
        v_hat = v4->z * bias_correction2;
        params4->z -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params4->z);

        // Fourth component
        m4->w = beta1 * m4->w + (1.0f - beta1) * g4->w;
        v4->w = beta2 * v4->w + (1.0f - beta2) * g4->w * g4->w;
        m_hat = m4->w * bias_correction1;
        v_hat = v4->w * bias_correction2;
        params4->w -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params4->w);
    }
}

void CUDABackend::device_adamw_step(float *params, float *g, float *m, float *v, int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step)
{
    size /= 4; // Adjust size for float4 processing

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    float bias_correction1 = 1.0f / (1.0f - powf(beta1, step));
    float bias_correction2 = 1.0f / (1.0f - powf(beta2, step));

    adamw_step_f32_v4_kernel<<<grid_size, block_size>>>(
        params,
        g,
        m,
        v,
        size,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
        weight_decay);

    CUDA_KERNEL_CHECK();
}