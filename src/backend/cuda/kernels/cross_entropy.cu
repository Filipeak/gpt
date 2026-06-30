#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

#define EPSILON 1e-8f

__global__ void cross_entropy_softmax_fused_backward_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ y_softmax,
    const int *__restrict__ tokens_labels,
    int batch_size,
    int seq_len,
    int vocab_size)
{
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y * blockDim.y + threadIdx.y;
    const int vocab = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch >= batch_size || seq >= seq_len || vocab >= vocab_size)
    {
        return;
    }

    const int label_idx = TENSOR_IDX_2D(batch, seq, seq_len);
    const int label = tokens_labels[label_idx];
    const int grad_idx = TENSOR_IDX_3D(batch, seq, vocab, seq_len, vocab_size);

    grad_x[grad_idx] = (y_softmax[grad_idx] - (vocab == label ? 1.0f : 0.0f)) / (float)(batch_size * seq_len);
}

void CUDABackend::device_cross_entropy_softmax_fused_backward(float *grad_x, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size)
{
    const dim3 blockDim(8, 8, 8);
    const dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x,
                       (seq_len + blockDim.y - 1) / blockDim.y,
                       (vocab_size + blockDim.z - 1) / blockDim.z);

    cross_entropy_softmax_fused_backward_kernel<<<gridDim, blockDim>>>(
        grad_x,
        y_softmax,
        tokens_labels,
        batch_size,
        seq_len,
        vocab_size);

    CUDA_DEBUG_SYNC();
}

__global__ void cross_entropy_compute_kernel(float *loss, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size * seq_len)
    {
        return;
    }

    const int batch_idx = row / seq_len;
    const int seq_idx = row % seq_len;
    const int label = tokens_labels[TENSOR_IDX_2D(batch_idx, seq_idx, seq_len)];

    float prob = y_softmax[TENSOR_IDX_3D(batch_idx, seq_idx, label, seq_len, vocab_size)];
    float log_prob = -logf(prob + EPSILON); // Add small epsilon to avoid log(0)

    log_prob /= (float)(batch_size * seq_len); // Normalize by total number of elements

    atomicAdd(loss, log_prob);
}

void CUDABackend::device_cross_entropy_loss(float *loss, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size)
{
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));

    const int block_size = 256;
    const int grid_size = (batch_size * seq_len + block_size - 1) / block_size;

    cross_entropy_compute_kernel<<<grid_size, block_size>>>(
        loss,
        y_softmax,
        tokens_labels,
        batch_size,
        seq_len,
        vocab_size);

    CUDA_DEBUG_SYNC();
}