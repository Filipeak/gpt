#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"
#include <float.h>

__global__ void softmax_forward_kernel(
    float *__restrict__ y,
    const float *__restrict__ x,
    int n,
    int d_model)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n)
    {
        return;
    }

    float max_val = -FLT_MAX;

    for (int i = 0; i < d_model; i++)
    {
        float val = x[TENSOR_IDX_2D(row, i, d_model)];

        if (val > max_val)
        {
            max_val = val;
        }
    }

    float sum_exp = 0.0f;

    for (int i = 0; i < d_model; i++)
    {
        float exp_val = expf(x[TENSOR_IDX_2D(row, i, d_model)] - max_val);

        y[TENSOR_IDX_2D(row, i, d_model)] = exp_val;
        sum_exp += exp_val;
    }

    for (int i = 0; i < d_model; i++)
    {
        y[TENSOR_IDX_2D(row, i, d_model)] /= sum_exp;
    }
}

void CUDABackend::device_softmax_forward(float *y, const float *x, int batch_size, int seq_len, int hidden_size)
{
    const int n = batch_size * seq_len;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    softmax_forward_kernel<<<grid_size, block_size>>>(
        y,
        x,
        n,
        hidden_size);

    CUDA_KERNEL_CHECK();
}

__global__ void softmax_backward_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ grad_y,
    const float *__restrict__ y,
    int n,
    int d_model)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n)
    {
        return;
    }

    float dot_product = 0.0f;

    for (int i = 0; i < d_model; i++)
    {
        const int idx = TENSOR_IDX_2D(row, i, d_model);
        dot_product += grad_y[idx] * y[idx];
    }

    for (int i = 0; i < d_model; i++)
    {
        const int idx = TENSOR_IDX_2D(row, i, d_model);
        grad_x[idx] = y[idx] * (grad_y[idx] - dot_product);
    }
}

void CUDABackend::device_softmax_backward(float *grad_x, const float *grad_y, const float *y, int batch_size, int seq_len, int size)
{
    const int n = batch_size * seq_len;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    softmax_backward_kernel<<<grid_size, block_size>>>(
        grad_x,
        grad_y,
        y,
        n,
        size);

    CUDA_KERNEL_CHECK();
}