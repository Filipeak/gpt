#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"
#include <float.h>

__global__ void multiheaded_softmax_masked_fused_kernel(
    float *__restrict__ x,
    int n,
    int seq_len)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n)
    {
        return;
    }

    const int q_idx = row % seq_len;

    float max_val = -FLT_MAX;

    for (int i = 0; i < seq_len; i++)
    {
        if (i > q_idx) // Mask future tokens
        {
            continue;
        }

        float val = x[TENSOR_IDX_2D(row, i, seq_len)];

        if (val > max_val)
        {
            max_val = val;
        }
    }

    float sum_exp = 0.0f;

    for (int i = 0; i < seq_len; i++)
    {
        if (i > q_idx) // Mask future tokens
        {
            continue;
        }

        sum_exp += expf(x[TENSOR_IDX_2D(row, i, seq_len)] - max_val);
    }

    for (int i = 0; i < seq_len; i++)
    {
        if (i > q_idx) // Mask future tokens
        {
            x[TENSOR_IDX_2D(row, i, seq_len)] = 0.0f;
        }
        else
        {
            x[TENSOR_IDX_2D(row, i, seq_len)] = expf(x[TENSOR_IDX_2D(row, i, seq_len)] - max_val) / sum_exp;
        }
    }
}

void CUDABackend::device_attention_forward(float *y, float *scores, const float *qkv, int batch_size, int seq_len, int hidden_size, int num_heads)
{
    const int head_size = hidden_size / num_heads;

    // Compute scores = (QK^T) / sqrt(d_k)
    const float alpha_scores = 1.0f / sqrtf((float)head_size);
    const float beta_scores = 0.0f;

    for (int head = 0; head < num_heads; head++)
    {
        const float *q_head = qkv + head * head_size;
        const float *k_head = qkv + hidden_size + head * head_size;
        float *scores_head = scores + head * seq_len * seq_len;

        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len, seq_len, head_size,
            &alpha_scores,
            k_head, 3 * hidden_size, seq_len * 3 * hidden_size,
            q_head, 3 * hidden_size, seq_len * 3 * hidden_size,
            &beta_scores,
            scores_head, seq_len, num_heads * seq_len * seq_len,
            batch_size));

        CUDA_KERNEL_CHECK();
    }

    // Normalize scores with softmax
    const int n = batch_size * num_heads * seq_len;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    multiheaded_softmax_masked_fused_kernel<<<grid_size, block_size>>>(
        scores,
        n,
        seq_len);

    CUDA_KERNEL_CHECK();

    // Final weighting
    const float alpha_final = 1.0f;
    const float beta_final = 0.0f;

    for (int head = 0; head < num_heads; head++)
    {
        const float *v_head = qkv + hidden_size * 2 + head * head_size;
        const float *scores_head = scores + head * seq_len * seq_len;
        float *out_head = y + head * head_size;

        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_size, seq_len, seq_len,
            &alpha_final,
            v_head, 3 * hidden_size, seq_len * 3 * hidden_size,
            scores_head, seq_len, num_heads * seq_len * seq_len,
            &beta_final,
            out_head, hidden_size, seq_len * hidden_size,
            batch_size));

        CUDA_KERNEL_CHECK();
    }
}

__global__ void softmax_backward_batched_kernel(
    float *__restrict__ grad,
    const float *__restrict__ y,
    int batch_size,
    int seq_len,
    int stride)
{
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch >= batch_size || seq >= seq_len)
    {
        return;
    }

    const int current_row_idx = batch * stride + seq * seq_len;

    float dot_product = 0.0f;

    for (int i = 0; i < seq_len; i++)
    {
        const int idx = current_row_idx + i;
        dot_product += grad[idx] * y[idx];
    }

    for (int i = 0; i < seq_len; i++)
    {
        const int idx = current_row_idx + i;
        grad[idx] = y[idx] * (grad[idx] - dot_product);
    }
}

void CUDABackend::device_attention_backward(float *grad_x, float *grad_softmax_cache, const float *grad_y, const float *qkv, const float *attn_scores, int batch_size, int seq_len, int hidden_size, int num_heads)
{
    const int head_size = hidden_size / num_heads;
    const float alpha = 1.0f;
    const float alpha_d_k = 1.0f / sqrtf((float)head_size);
    const float beta = 0.0f;

    for (int head = 0; head < num_heads; head++)
    {
        // Store intermediate values
        const float *grad_y_head = grad_y + head * head_size;
        const float *attn_scores_head = attn_scores + head * seq_len * seq_len;
        const float *q_head = qkv + hidden_size * 0 + head * head_size;
        const float *k_head = qkv + hidden_size * 1 + head * head_size;
        const float *v_head = qkv + hidden_size * 2 + head * head_size;
        float *grad_softmax_cache_head = grad_softmax_cache + head * seq_len * seq_len;

        // Compute grad_softmax_cache = grad_y * V^T
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len, seq_len, head_size,
            &alpha,
            v_head, 3 * hidden_size, seq_len * 3 * hidden_size,
            grad_y_head, hidden_size, seq_len * hidden_size,
            &beta,
            grad_softmax_cache_head, seq_len, num_heads * seq_len * seq_len,
            batch_size));

        CUDA_KERNEL_CHECK();

        // Compute softmax backward for the given cache
        const dim3 block_size(16, 16);
        const dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, (seq_len + block_size.y - 1) / block_size.y);

        softmax_backward_batched_kernel<<<grid_size, block_size>>>(
            grad_softmax_cache_head,
            attn_scores_head,
            batch_size,
            seq_len,
            num_heads * seq_len * seq_len);

        CUDA_KERNEL_CHECK();

        // Compute grad_q = grad_softmax_cache * K / sqrt(d_k)
        float *grad_x_head_q = grad_x + hidden_size * 0 + head * head_size;

        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_size, seq_len, seq_len,
            &alpha_d_k,
            k_head, 3 * hidden_size, seq_len * 3 * hidden_size,
            grad_softmax_cache_head, seq_len, num_heads * seq_len * seq_len,
            &beta,
            grad_x_head_q, 3 * hidden_size, seq_len * 3 * hidden_size,
            batch_size));

        CUDA_KERNEL_CHECK();

        // Compute grad_k = grad_softmax_cache^T * Q / sqrt(d_k)
        float *grad_x_head_k = grad_x + hidden_size * 1 + head * head_size;

        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            head_size, seq_len, seq_len,
            &alpha_d_k,
            q_head, 3 * hidden_size, seq_len * 3 * hidden_size,
            grad_softmax_cache_head, seq_len, num_heads * seq_len * seq_len,
            &beta,
            grad_x_head_k, 3 * hidden_size, seq_len * 3 * hidden_size,
            batch_size));

        CUDA_KERNEL_CHECK();

        // Compute grad_v = attn_scores^T * grad_y
        float *grad_x_head_v = grad_x + hidden_size * 2 + head * head_size;

        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            head_size, seq_len, seq_len,
            &alpha,
            grad_y_head, hidden_size, seq_len * hidden_size,
            attn_scores_head, seq_len, num_heads * seq_len * seq_len,
            &beta,
            grad_x_head_v, 3 * hidden_size, seq_len * 3 * hidden_size,
            batch_size));

        CUDA_KERNEL_CHECK();
    }
}