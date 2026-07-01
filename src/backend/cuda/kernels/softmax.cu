#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"
#include <float.h>

__global__ void softmax_forward_kernel(
    float *__restrict__ y,
    const float *__restrict__ x,
    int n,
    int vocab_size,
    int vocab_size_padded)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n)
    {
        return;
    }

    float max_val = -FLT_MAX;

    for (int i = 0; i < vocab_size; i++)
    {
        float val = x[TENSOR_IDX_2D(row, i, vocab_size_padded)];

        if (val > max_val)
        {
            max_val = val;
        }
    }

    float sum_exp = 0.0f;

    for (int i = 0; i < vocab_size; i++)
    {
        float exp_val = expf(x[TENSOR_IDX_2D(row, i, vocab_size_padded)] - max_val);

        y[TENSOR_IDX_2D(row, i, vocab_size_padded)] = exp_val;
        sum_exp += exp_val;
    }

    for (int i = 0; i < vocab_size; i++)
    {
        y[TENSOR_IDX_2D(row, i, vocab_size_padded)] /= sum_exp;
    }

    // Set the padded values to zero
    for (int i = vocab_size; i < vocab_size_padded; i++)
    {
        y[TENSOR_IDX_2D(row, i, vocab_size_padded)] = 0.0f;
    }
}

void CUDABackend::device_softmax_forward(float *y, const float *x, int batch_size, int seq_len, int vocab_size, int vocab_size_padded)
{
    const int n = batch_size * seq_len;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    softmax_forward_kernel<<<grid_size, block_size>>>(
        y,
        x,
        n,
        vocab_size,
        vocab_size_padded);

    CUDA_KERNEL_CHECK();
}

__global__ void softmax_backward_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ grad_y,
    const float *__restrict__ y,
    int n,
    int vocab_size,
    int vocab_size_padded)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n)
    {
        return;
    }

    float dot_product = 0.0f;

    for (int i = 0; i < vocab_size; i++)
    {
        const int idx = TENSOR_IDX_2D(row, i, vocab_size_padded);
        dot_product += grad_y[idx] * y[idx];
    }

    for (int i = 0; i < vocab_size; i++)
    {
        const int idx = TENSOR_IDX_2D(row, i, vocab_size_padded);
        grad_x[idx] = y[idx] * (grad_y[idx] - dot_product);
    }
}

void CUDABackend::device_softmax_backward(float *grad_x, const float *grad_y, const float *y, int batch_size, int seq_len, int vocab_size, int vocab_size_padded)
{
    const int n = batch_size * seq_len;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    softmax_backward_kernel<<<grid_size, block_size>>>(
        grad_x,
        grad_y,
        y,
        n,
        vocab_size,
        vocab_size_padded);

    CUDA_KERNEL_CHECK();
}