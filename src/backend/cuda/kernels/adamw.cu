#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

__global__ void adamw_step_kernel(
    float *params,
    float *g,
    float *m,
    float *v,
    int size,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps,
    float weight_decay)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g[idx] * g[idx];

        float m_hat = m[idx] * bias_correction1;
        float v_hat = v[idx] * bias_correction2;

        params[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params[idx]);
    }
}

void CUDABackend::device_adamw_step(float *params, float *g, float *m, float *v, int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step)
{
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    float bias_correction1 = 1.0f / (1.0f - powf(beta1, step));
    float bias_correction2 = 1.0f / (1.0f - powf(beta2, step));

    adamw_step_kernel<<<grid_size, block_size>>>(
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
}