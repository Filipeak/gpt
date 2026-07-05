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

        float4 p_val = *params4;
        float4 g_val = *g4;
        float4 m_val = *m4;
        float4 v_val = *v4;

        float m_hat, v_hat;

        // First component
        m_val.x = beta1 * m_val.x + (1.0f - beta1) * g_val.x;
        v_val.x = beta2 * v_val.x + (1.0f - beta2) * g_val.x * g_val.x;
        m_hat = m_val.x * bias_correction1;
        v_hat = v_val.x * bias_correction2;
        p_val.x -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_val.x);

        // Second component
        m_val.y = beta1 * m_val.y + (1.0f - beta1) * g_val.y;
        v_val.y = beta2 * v_val.y + (1.0f - beta2) * g_val.y * g_val.y;
        m_hat = m_val.y * bias_correction1;
        v_hat = v_val.y * bias_correction2;
        p_val.y -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_val.y);

        // Third component
        m_val.z = beta1 * m_val.z + (1.0f - beta1) * g_val.z;
        v_val.z = beta2 * v_val.z + (1.0f - beta2) * g_val.z * g_val.z;
        m_hat = m_val.z * bias_correction1;
        v_hat = v_val.z * bias_correction2;
        p_val.z -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_val.z);

        // Fourth component
        m_val.w = beta1 * m_val.w + (1.0f - beta1) * g_val.w;
        v_val.w = beta2 * v_val.w + (1.0f - beta2) * g_val.w * g_val.w;
        m_hat = m_val.w * bias_correction1;
        v_hat = v_val.w * bias_correction2;
        p_val.w -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_val.w);

        // Assign updated values back to global memory
        *params4 = p_val;
        *m4 = m_val;
        *v4 = v_val;
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