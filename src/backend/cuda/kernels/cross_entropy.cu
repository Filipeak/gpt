#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

#define EPSILON 1e-8f

__global__ void cross_entropy_softmax_fused_backward_f32_v4_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ y_softmax,
    const int *__restrict__ tokens_labels,
    int num_rows,
    int vocab_size,
    int vocab_size_padded,
    float scale)
{
    const int row = blockIdx.x;

    if (row >= num_rows)
    {
        return;
    }

    const int label = tokens_labels[row];

    // Get pointers to current rows
    const float4 *y4_row = (const float4 *)(y_softmax + row * vocab_size_padded);
    float4 *grad4_row = (float4 *)(grad_x + row * vocab_size_padded);

    // First, handle the safe float4s (those that are fully within the vocab_size)
    const int num_float4_safe = vocab_size / 4;
    for (int i = threadIdx.x; i < num_float4_safe; i += blockDim.x)
    {
        float4 val = y4_row[i];
        float4 result;
        const int base_col = i * 4;

        result.x = (val.x - (float)(label == base_col + 0)) * scale;
        result.y = (val.y - (float)(label == base_col + 1)) * scale;
        result.z = (val.z - (float)(label == base_col + 2)) * scale;
        result.w = (val.w - (float)(label == base_col + 3)) * scale;

        grad4_row[i] = result;
    }

    // Second, handle the remaining float4s that may go beyond vocab_size
    const int num_float4_padded = vocab_size_padded / 4;
    for (int i = num_float4_safe + threadIdx.x; i < num_float4_padded; i += blockDim.x)
    {
        float4 val = y4_row[i];
        float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int base_col = i * 4;

        if (base_col + 0 < vocab_size)
            result.x = (val.x - (float)(label == base_col + 0)) * scale;
        if (base_col + 1 < vocab_size)
            result.y = (val.y - (float)(label == base_col + 1)) * scale;
        if (base_col + 2 < vocab_size)
            result.z = (val.z - (float)(label == base_col + 2)) * scale;
        if (base_col + 3 < vocab_size)
            result.w = (val.w - (float)(label == base_col + 3)) * scale;

        grad4_row[i] = result;
    }
}

void CUDABackend::device_cross_entropy_softmax_fused_backward(float *grad_x, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size, int vocab_size_padded)
{
    const int n = batch_size * seq_len;
    const int block_size = 1024;
    const int grid_size = n;

    cross_entropy_softmax_fused_backward_f32_v4_kernel<<<grid_size, block_size>>>(
        grad_x,
        y_softmax,
        tokens_labels,
        n,
        vocab_size,
        vocab_size_padded,
        1.0f / (float)n);

    CUDA_KERNEL_CHECK();
}

// TODO: For larger sequences use block reduction + atomics
__global__ void cross_entropy_compute_kernel(
    float *__restrict__ loss,
    const float *__restrict__ y_softmax,
    const int *__restrict__ tokens_labels,
    int num_rows,
    int vocab_size_padded,
    float scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= num_rows)
    {
        return;
    }

    const int label = tokens_labels[row];

    float prob = y_softmax[row * vocab_size_padded + label];
    float log_prob = -logf(prob + EPSILON); // Add small epsilon to avoid log(0)

    log_prob *= scale;

    atomicAdd(loss, log_prob);
}

void CUDABackend::device_cross_entropy_loss(float *loss, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size_padded)
{
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));

    const int n = batch_size * seq_len;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cross_entropy_compute_kernel<<<grid_size, block_size>>>(
        loss,
        y_softmax,
        tokens_labels,
        n,
        vocab_size_padded,
        1.0f / (float)n);

    CUDA_KERNEL_CHECK();
}