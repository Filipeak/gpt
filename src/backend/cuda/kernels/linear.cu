#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

// TODO: Cache the tanh and gelu values to avoid recomputation in backward pass
// TODO: Consider fusing entire GEMM + bias + activation into a single kernel for better performance
__global__ void add_bias_and_activate_fused_kernel(
    float *__restrict__ y,
    float *__restrict__ act,
    const float *__restrict__ bias,
    bool activation,
    int n_rows,
    int n_cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

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
    const dim3 blockDim(32, 8);
    const dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    add_bias_and_activate_fused_kernel<<<gridDim, blockDim>>>(
        y,
        act,
        b,
        activation,
        n,
        output_size);

    CUDA_KERNEL_CHECK();
}

// TODO: Consider using cublasSgemv instead of custom kernel
// TODO: Use float4 for vectorized access to improve performance
__global__ void linear_bias_gradient_kernel(
    float *__restrict__ grad_b,
    const float *__restrict__ grad_y,
    int n_rows,
    int n_cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Compute local sum for this thread
    float local_sum = 0.0f;

    if (col < n_cols)
    {
        for (int row = threadIdx.y; row < n_rows; row += blockDim.y)
        {
            local_sum += grad_y[TENSOR_IDX_2D(row, col, n_cols)];
        }
    }

    // Use shared memory to accumulate the local sums from all threads in the block
    extern __shared__ float smem[];
    float *shared_grad_b = smem;

    shared_grad_b[tid] = local_sum;

    __syncthreads();

    // Reduce the shared memory values to compute the final gradient for this column
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.y < stride)
        {
            shared_grad_b[tid] += shared_grad_b[tid + stride * blockDim.x];
        }

        __syncthreads();
    }

    // Finally write the accumulated gradients for bias to global memory
    if (threadIdx.y == 0 && col < n_cols)
    {
        grad_b[col] += shared_grad_b[threadIdx.x];
    }
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
    const dim3 blockDim(32, 32);
    const dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x);
    const size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(float);

    linear_bias_gradient_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        grad_b,
        grad_y,
        batch_size * seq_len,
        output_size);

    CUDA_KERNEL_CHECK();
}

// TODO: Use float4 for vectorized access to improve performance
__global__ void activation_backward_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ grad_y,
    const float *__restrict__ x,
    int n_rows,
    int n_cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

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
    const dim3 blockDim(32, 8);
    const dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    activation_backward_kernel<<<gridDim, blockDim>>>(
        grad_x,
        grad_y,
        x,
        n,
        output_size);

    CUDA_KERNEL_CHECK();
}