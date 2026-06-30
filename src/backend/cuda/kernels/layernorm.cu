#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

#define EPSILON 1e-5f

__global__ void layernorm_forward_kernel(
    float *__restrict__ y,
    float *__restrict__ means,
    float *__restrict__ vars,
    const float *__restrict__ x,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    int n,
    int d_model)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n)
    {
        return;
    }

    // Mean calculation
    float mean = 0.0f;

    for (int channel = 0; channel < d_model; channel++)
    {
        mean += x[TENSOR_IDX_2D(row, channel, d_model)];
    }

    mean /= d_model;

    means[row] = mean;

    // Variance calculation
    float variance = 0.0f;

    for (int channel = 0; channel < d_model; channel++)
    {
        float diff = x[TENSOR_IDX_2D(row, channel, d_model)] - mean;

        variance += diff * diff;
    }

    variance /= d_model;

    vars[row] = variance;

    // Final normalization
    for (int channel = 0; channel < d_model; channel++)
    {
        float normalized = (x[TENSOR_IDX_2D(row, channel, d_model)] - mean) * rsqrtf(variance + EPSILON);

        y[TENSOR_IDX_2D(row, channel, d_model)] = normalized * weight[channel] + bias[channel];
    }
}

void CUDABackend::device_layernorm_forward(float *y, float *means, float *vars, const float *x, const float *gamma, const float *beta, int batch_size, int seq_len, int hidden_size)
{
    const int n = batch_size * seq_len;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    layernorm_forward_kernel<<<grid_size, block_size>>>(
        y,
        means,
        vars,
        x,
        gamma,
        beta,
        n,
        hidden_size);

    CUDA_DEBUG_SYNC();
}

__global__ void layernorm_backward_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ grad_y,
    const float *__restrict__ x,
    const float *__restrict__ means,
    const float *__restrict__ vars,
    const float *__restrict__ gamma,
    int n,
    int d_model)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n)
    {
        return;
    }

    float mean = means[row];
    float variance = vars[row];
    float inv_std = rsqrtf(variance + EPSILON);

    float sum1 = 0.0f;
    float sum2 = 0.0f;

    for (int channel = 0; channel < d_model; channel++)
    {
        float normalized = (x[TENSOR_IDX_2D(row, channel, d_model)] - mean) * inv_std;
        float dy = grad_y[TENSOR_IDX_2D(row, channel, d_model)];

        sum1 += dy * gamma[channel];
        sum2 += dy * gamma[channel] * normalized;
    }

    sum1 /= d_model;
    sum2 /= d_model;

    for (int channel = 0; channel < d_model; channel++)
    {
        float normalized = (x[TENSOR_IDX_2D(row, channel, d_model)] - mean) * inv_std;
        float dy = grad_y[TENSOR_IDX_2D(row, channel, d_model)];

        grad_x[TENSOR_IDX_2D(row, channel, d_model)] = inv_std * (dy * gamma[channel] - sum1 - normalized * sum2);
    }
}

__global__ void layernorm_weights_gradients_fused_kernel(
    float *__restrict__ grad_gamma,
    float *__restrict__ grad_beta,
    const float *__restrict__ grad_y,
    const float *__restrict__ x,
    const float *__restrict__ means,
    const float *__restrict__ vars,
    const float *__restrict__ gamma,
    int n,
    int d_model)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;

    if (channel >= d_model)
    {
        return;
    }

    float grad_gamma_sum = 0.0f;
    float grad_beta_sum = 0.0f;

    for (int row = 0; row < n; row++)
    {
        float normalized = (x[TENSOR_IDX_2D(row, channel, d_model)] - means[row]) * rsqrtf(vars[row] + EPSILON);
        float dy = grad_y[TENSOR_IDX_2D(row, channel, d_model)];

        grad_gamma_sum += dy * normalized;
        grad_beta_sum += dy;
    }

    grad_gamma[channel] += grad_gamma_sum;
    grad_beta[channel] += grad_beta_sum;
}

void CUDABackend::device_layernorm_backward(float *grad_x, float *grad_gamma, float *grad_beta, const float *grad_y, const float *x, const float *means, const float *vars, const float *gamma, int batch_size, int seq_len, int hidden_size)
{
    const int n = batch_size * seq_len;
    const int block_size_backward = 256;
    const int grid_size_backward = (n + block_size_backward - 1) / block_size_backward;

    layernorm_backward_kernel<<<grid_size_backward, block_size_backward>>>(
        grad_x,
        grad_y,
        x,
        means,
        vars,
        gamma,
        n,
        hidden_size);

    CUDA_DEBUG_SYNC();

    const int block_size_weights = 256;
    const int grid_size_weights = (hidden_size + block_size_weights - 1) / block_size_weights;

    layernorm_weights_gradients_fused_kernel<<<grid_size_weights, block_size_weights>>>(
        grad_gamma,
        grad_beta,
        grad_y,
        x,
        means,
        vars,
        gamma,
        n,
        hidden_size);

    CUDA_DEBUG_SYNC();
}