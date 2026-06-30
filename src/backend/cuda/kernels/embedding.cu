#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

__global__ void embedding_forward_kernel(
    float *__restrict__ y,
    const float *__restrict__ wte,
    const float *__restrict__ wpe,
    const int *__restrict__ input_tokens,
    int batch_size,
    int seq_len,
    int d_model)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (b >= batch_size || t >= seq_len || c >= d_model)
    {
        return;
    }

    const int token_id = input_tokens[b * seq_len + t];
    const int y_idx = TENSOR_IDX_3D(b, t, c, seq_len, d_model);
    const int wte_idx = TENSOR_IDX_2D(token_id, c, d_model);
    const int wpe_idx = TENSOR_IDX_2D(t, c, d_model);

    y[y_idx] = wte[wte_idx] + wpe[wpe_idx];
}

void CUDABackend::device_embedding_forward(float *y, const float *wte, const float *wpe, const int *input_tokens, int batch_size, int seq_len, int hidden_size)
{
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y, (hidden_size + blockDim.z - 1) / blockDim.z);

    embedding_forward_kernel<<<gridDim, blockDim>>>(
        y,
        wte,
        wpe,
        input_tokens,
        batch_size,
        seq_len,
        hidden_size);

    CUDA_DEBUG_SYNC();
}

void CUDABackend::device_unembedding_forward(float *y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        vocab_size, batch_size * seq_len, hidden_size,
        &alpha,
        wte, hidden_size,
        x, hidden_size,
        &beta,
        y, vocab_size));

    CUDA_DEBUG_SYNC();
}

__global__ void embedding_backward_kernel(
    float *__restrict__ grad_wte,
    float *__restrict__ grad_wpe,
    const float *__restrict__ grad_y,
    const int *__restrict__ input_tokens,
    int batch_size,
    int seq_len,
    int hidden_size)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (b >= batch_size || t >= seq_len || c >= hidden_size)
    {
        return;
    }

    const int token_id = input_tokens[b * seq_len + t];
    const int grad_y_idx = TENSOR_IDX_3D(b, t, c, seq_len, hidden_size);
    const int grad_wte_idx = TENSOR_IDX_2D(token_id, c, hidden_size);
    const int grad_wpe_idx = TENSOR_IDX_2D(t, c, hidden_size);

    atomicAdd(&grad_wte[grad_wte_idx], grad_y[grad_y_idx]);
    atomicAdd(&grad_wpe[grad_wpe_idx], grad_y[grad_y_idx]);
}

void CUDABackend::device_embedding_backward(float *grad_wte, float *grad_wpe, const float *grad_y, const int *input_tokens, int batch_size, int seq_len, int hidden_size)
{
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y, (hidden_size + blockDim.z - 1) / blockDim.z);

    embedding_backward_kernel<<<gridDim, blockDim>>>(
        grad_wte,
        grad_wpe,
        grad_y,
        input_tokens,
        batch_size,
        seq_len,
        hidden_size);

    CUDA_DEBUG_SYNC();
}

void CUDABackend::device_unembedding_backward(float *grad_x, float *grad_wte, const float *grad_y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute grad_x = grad_y @ wte
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size * seq_len, vocab_size,
        &alpha,
        wte, hidden_size,
        grad_y, vocab_size,
        &beta,
        grad_x, hidden_size));

    CUDA_DEBUG_SYNC();

    // Compute grad_wte += grad_y^T @ x
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size, vocab_size, batch_size * seq_len,
        &alpha,
        x, hidden_size,
        grad_y, vocab_size,
        &alpha, // Use alpha to accumulate into grad_wte
        grad_wte, hidden_size));

    CUDA_DEBUG_SYNC();
}