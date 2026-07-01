#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

__global__ void add_bias_and_activate_fused_kernel(
    float *__restrict__ y,
    float *__restrict__ act,
    const float *__restrict__ bias,
    bool activation,
    int n_rows,
    int n_cols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n_rows || col >= n_cols)
    {
        return;
    }

    const int cur_idx = TENSOR_IDX_2D(row, col, n_cols);

    y[cur_idx] += bias[col];

    if (activation)
    {
        float val = y[cur_idx];
        float gelu = 0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));

        act[cur_idx] = gelu;
    }
}

void CUDABackend::device_linear_activation_fused_forward(float *y, float *act, const float *x, const float *w, const float *b, bool activation, int batch_size, int seq_len, int input_size, int output_size)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        output_size, batch_size * seq_len, input_size,
        &alpha,
        w, output_size,
        x, input_size,
        &beta,
        y, output_size));

    CUDA_KERNEL_CHECK();

    const int n = batch_size * seq_len;
    const dim3 blockDim(16, 16);
    const dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (output_size + blockDim.y - 1) / blockDim.y);

    add_bias_and_activate_fused_kernel<<<gridDim, blockDim>>>(
        y,
        act,
        b,
        activation,
        n,
        output_size);

    CUDA_KERNEL_CHECK();
}

__global__ void linear_bias_gradient_kernel(
    float *__restrict__ grad_b,
    const float *__restrict__ grad_y,
    int n_rows,
    int n_cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= n_cols)
    {
        return;
    }

    float sum = 0.0f;

    for (int row = 0; row < n_rows; ++row)
    {
        const int cur_idx = TENSOR_IDX_2D(row, col, n_cols);

        sum += grad_y[cur_idx];
    }

    grad_b[col] += sum;
}

void CUDABackend::device_linear_backward(float *grad_x, float *grad_w, float *grad_b, const float *grad_y, const float *x, const float *w, int batch_size, int seq_len, int input_size, int output_size)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // grad_x = grad_y @ w^T
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        input_size, batch_size * seq_len, output_size,
        &alpha,
        w, output_size,
        grad_y, output_size,
        &beta,
        grad_x, input_size));

    CUDA_KERNEL_CHECK();

    // grad_w += x^T @ grad_y
    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        output_size, input_size, batch_size * seq_len,
        &alpha,
        grad_y, output_size,
        x, input_size,
        &alpha, // Using alpha instead of beta to accumulate into
        grad_w, output_size));

    CUDA_KERNEL_CHECK();

    // grad_b += sum(grad_y, axis=0)
    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    linear_bias_gradient_kernel<<<grid_size, block_size>>>(
        grad_b,
        grad_y,
        batch_size * seq_len,
        output_size);

    CUDA_KERNEL_CHECK();
}

__global__ void activation_backward_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ grad_y,
    const float *__restrict__ x,
    int n_rows,
    int n_cols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n_rows || col >= n_cols)
    {
        return;
    }

    const int cur_idx = TENSOR_IDX_2D(row, col, n_cols);

    float val = x[cur_idx];
    float grad_val = grad_y[cur_idx];

    // Derivative of GELU activation
    float tanh_out = tanhf(0.79788456f * (val + 0.044715f * val * val * val));
    float gelu_derivative = 0.5f * (1.0f + tanh_out) + 0.5f * val * (1.0f - tanh_out * tanh_out) * (0.79788456f + 0.107032f * val * val);

    grad_x[cur_idx] = grad_val * gelu_derivative;
}

void CUDABackend::device_activation_backward(float *grad_x, const float *grad_y, const float *x, int batch_size, int seq_len, int output_size)
{
    const int n = batch_size * seq_len;
    const dim3 blockDim(16, 16);
    const dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (output_size + blockDim.y - 1) / blockDim.y);

    activation_backward_kernel<<<gridDim, blockDim>>>(
        grad_x,
        grad_y,
        x,
        n,
        output_size);

    CUDA_KERNEL_CHECK();
}