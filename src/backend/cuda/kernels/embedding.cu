#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

__global__ void embedding_forward_f32_kernel(
    float *__restrict__ y,
    const float *__restrict__ wte,
    const float *__restrict__ wpe,
    const int *__restrict__ input_tokens,
    int n,
    int d_model)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= d_model || row >= n)
    {
        return;
    }

    const int token_id = input_tokens[row];
    const int y_idx = row * d_model + col;
    const int wte_idx = token_id * d_model + col;
    const int wpe_idx = row * d_model + col;

    y[y_idx] = wte[wte_idx] + wpe[wpe_idx];
}

void CUDABackend::device_embedding_forward(float *y, const float *wte, const float *wpe, const int *input_tokens, int batch_size, int seq_len, int hidden_size)
{
    const int n = batch_size * seq_len;

    dim3 blockDim(32, 32);
    dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    embedding_forward_f32_kernel<<<gridDim, blockDim>>>(
        y,
        wte,
        wpe,
        input_tokens,
        n,
        hidden_size);

    CUDA_KERNEL_CHECK();
}

void CUDABackend::device_unembedding_forward(float *y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size_padded)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        vocab_size_padded, batch_size * seq_len, hidden_size,
        &alpha,
        wte, hidden_size,
        x, hidden_size,
        &beta,
        y, vocab_size_padded));

    CUDA_KERNEL_CHECK();
}

__global__ void wpe_backward_kernel(
    float *__restrict__ grad_wpe,
    const float *__restrict__ grad_y,
    int batch_size,
    int seq_len,
    int hidden_size)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= hidden_size || t >= seq_len)
    {
        return;
    }

    // Sum the same positions for all batches
    float sum = 0.0f;

    for (int b = 0; b < batch_size; ++b)
    {
        sum += grad_y[TENSOR_IDX_3D(b, t, c, seq_len, hidden_size)];
    }

    // Add final sum
    grad_wpe[TENSOR_IDX_2D(t, c, hidden_size)] += sum;
}

__global__ void wte_backward_kernel(
    float *__restrict__ grad_wte,
    const float *__restrict__ grad_y,
    const int *__restrict__ input_tokens,
    int n,
    int hidden_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= hidden_size || row >= n)
    {
        return;
    }

    const int token_id = input_tokens[row];

    atomicAdd(&grad_wte[TENSOR_IDX_2D(token_id, col, hidden_size)], grad_y[TENSOR_IDX_2D(row, col, hidden_size)]);
}

void CUDABackend::device_embedding_backward(float *grad_wte, float *grad_wpe, const float *grad_y, const int *input_tokens, int batch_size, int seq_len, int hidden_size)
{
    dim3 wpe_blockDim(32, 8);
    dim3 wpe_gridDim((hidden_size + wpe_blockDim.x - 1) / wpe_blockDim.x, (seq_len + wpe_blockDim.y - 1) / wpe_blockDim.y);

    wpe_backward_kernel<<<wpe_gridDim, wpe_blockDim>>>(
        grad_wpe,
        grad_y,
        batch_size,
        seq_len,
        hidden_size);

    CUDA_KERNEL_CHECK();

    const int n = batch_size * seq_len;
    dim3 wte_blockDim(32, 8);
    dim3 wte_gridDim((hidden_size + wte_blockDim.x - 1) / wte_blockDim.x, (n + wte_blockDim.y - 1) / wte_blockDim.y);

    wte_backward_kernel<<<wte_gridDim, wte_blockDim>>>(
        grad_wte,
        grad_y,
        input_tokens,
        n,
        hidden_size);

    CUDA_KERNEL_CHECK();
}

void CUDABackend::device_unembedding_backward(float *grad_x, float *grad_wte, const float *grad_y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size_padded)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute grad_x = grad_y @ wte
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size * seq_len, vocab_size_padded,
        &alpha,
        wte, hidden_size,
        grad_y, vocab_size_padded,
        &beta,
        grad_x, hidden_size));

    CUDA_KERNEL_CHECK();

    // Compute grad_wte += grad_y^T @ x
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size, vocab_size_padded, batch_size * seq_len,
        &alpha,
        x, hidden_size,
        grad_y, vocab_size_padded,
        &alpha, // Use alpha to accumulate into grad_wte
        grad_wte, hidden_size));

    CUDA_KERNEL_CHECK();
}