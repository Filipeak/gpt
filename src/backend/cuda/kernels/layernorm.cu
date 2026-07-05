#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"
#include <math.h>

#define EPSILON 1e-5f

struct WelfordsData
{
    int count;
    float mean;
    float m2;
};

__inline__ __device__ void warpReduceWelfords(
    WelfordsData &data)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        WelfordsData other_data;
        other_data.count = __shfl_down_sync(0xffffffff, data.count, offset);
        other_data.mean = __shfl_down_sync(0xffffffff, data.mean, offset);
        other_data.m2 = __shfl_down_sync(0xffffffff, data.m2, offset);

        if (other_data.count > 0)
        {
            int total_count = data.count + other_data.count;
            float delta = other_data.mean - data.mean;
            data.mean += delta * other_data.count / total_count;
            data.m2 += other_data.m2 + delta * delta * data.count * other_data.count / total_count;
            data.count = total_count;
        }
    }
}

__global__ void layernorm_residual_fused_forward_f32_v4_kernel(
    float *__restrict__ y,
    float *__restrict__ means,
    float *__restrict__ vars,
    float *__restrict__ x,
    const float *__restrict__ residual,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    int n,
    int d_model)
{
    const int thread_id = threadIdx.x;
    const int row_idx = blockIdx.x;
    const int warp_count = blockDim.x / warpSize;
    const int warp_id = thread_id / warpSize;
    const int lane_id = thread_id % warpSize;

    if (row_idx >= n)
    {
        return;
    }

    // Use float4 to process 4 elements at a time for better memory coalescing and performance (assuming d_model is a multiple of 4 and memory alignment allows it)
    float4 *x4 = (float4 *)(x + row_idx * d_model);
    const float4 *residual4 = residual ? (const float4 *)(residual + row_idx * d_model) : nullptr;
    float4 *y4 = (float4 *)(y + row_idx * d_model);
    int num_float4 = d_model / 4;

    // Compute local mean and variance using Welford's algorithm
    WelfordsData local_welfords_data;
    local_welfords_data.count = 0;
    local_welfords_data.mean = 0.0f;
    local_welfords_data.m2 = 0.0f;

    for (int i = thread_id; i < num_float4; i += blockDim.x)
    {
        float4 val4 = x4[i];

        // Add residual connection
        float4 residual_val4 = residual4 ? residual4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        val4.x += residual_val4.x; 
        val4.y += residual_val4.y;
        val4.z += residual_val4.z;
        val4.w += residual_val4.w;
        x4[i] = val4; // Store the original x values for backward passs

        float vals[4] = {val4.x, val4.y, val4.z, val4.w};

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float v = vals[j];
            local_welfords_data.count += 1;
            float delta = v - local_welfords_data.mean;
            local_welfords_data.mean += delta / local_welfords_data.count;
            float delta2 = v - local_welfords_data.mean;
            local_welfords_data.m2 += delta * delta2;
        }
    }

    // Warp reduction
    warpReduceWelfords(local_welfords_data);

    // Block reduction (max block size is 1024 threads, so 32 warps)
    __shared__ WelfordsData shared_data[32];

    if (lane_id == 0)
    {
        shared_data[warp_id] = local_welfords_data;
    }

    __syncthreads();

    // Final reduction by the first warp
    __shared__ float shared_mean;
    __shared__ float shared_inv_std;

    if (warp_id == 0)
    {
        local_welfords_data = lane_id < warp_count ? shared_data[lane_id] : WelfordsData{0, 0.0f, 0.0f};

        warpReduceWelfords(local_welfords_data);

        if (lane_id == 0)
        {
            means[row_idx] = local_welfords_data.mean;
            vars[row_idx] = rsqrtf(local_welfords_data.m2 / local_welfords_data.count + EPSILON); // Store inverse std for normalization

            // Broadcast mean and variance to all threads in the block
            shared_mean = means[row_idx];
            shared_inv_std = vars[row_idx];
        }
    }

    __syncthreads();

    // Compute normalized output
    for (int i = thread_id; i < num_float4; i += blockDim.x)
    {
        float4 val4 = x4[i];
        float4 result;

        result.x = (val4.x - shared_mean) * shared_inv_std * weight[i * 4 + 0] + bias[i * 4 + 0];
        result.y = (val4.y - shared_mean) * shared_inv_std * weight[i * 4 + 1] + bias[i * 4 + 1];
        result.z = (val4.z - shared_mean) * shared_inv_std * weight[i * 4 + 2] + bias[i * 4 + 2];
        result.w = (val4.w - shared_mean) * shared_inv_std * weight[i * 4 + 3] + bias[i * 4 + 3];

        y4[i] = result;
    }
}

void CUDABackend::device_layernorm_residual_fused_forward(float *y, float *means, float *vars, float *x, const float *residual, const float *gamma, const float *beta, int batch_size, int seq_len, int hidden_size)
{
    const int n = batch_size * seq_len;
    const int block_size = min(1024, hidden_size / 4);

    layernorm_residual_fused_forward_f32_v4_kernel<<<n, block_size>>>(
        y,
        means,
        vars,
        x,
        residual,
        gamma,
        beta,
        n,
        hidden_size);

    CUDA_KERNEL_CHECK();
}

__inline__ __device__ void warpReduceLayernormSums(
    float &sum1,
    float &sum2)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
}

__global__ void layernorm_residual_fused_backward_f32_v4_kernel(
    float *__restrict__ grad_x,
    float *__restrict__ grad_gamma,
    float *__restrict__ grad_beta,
    const float *__restrict__ grad_y,
    const float *__restrict__ grad_residual,
    const float *__restrict__ x,
    const float *__restrict__ means,
    const float *__restrict__ vars,
    const float *__restrict__ gamma,
    int n,
    int d_model)
{
    const int thread_id = threadIdx.x;
    const int row_idx = blockIdx.x;
    const int warp_count = blockDim.x / warpSize;
    const int warp_id = thread_id / warpSize;
    const int lane_id = thread_id % warpSize;

    if (row_idx >= n)
    {
        return;
    }

    // Use float4 for vectorized access
    const float4 *x4 = (const float4 *)(x + row_idx * d_model);
    const float4 *grad_y4 = (const float4 *)(grad_y + row_idx * d_model);
    const float4 *grad_residual4 = grad_residual ? (const float4 *)(grad_residual + row_idx * d_model) : nullptr;
    int num_float4 = d_model / 4;

    // Cache mean and variance for this row
    float mean = means[row_idx];
    float inv_std = vars[row_idx];

    // Compute local sums
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    for (int i = thread_id; i < num_float4; i += blockDim.x)
    {
        float4 x_val = x4[i];
        float4 dy_val = grad_y4[i];

        float normalized[4] = {
            (x_val.x - mean) * inv_std,
            (x_val.y - mean) * inv_std,
            (x_val.z - mean) * inv_std,
            (x_val.w - mean) * inv_std,
        };
        float dy[4] = {dy_val.x, dy_val.y, dy_val.z, dy_val.w};

        for (int j = 0; j < 4; j++)
        {
            int channel = i * 4 + j;

            sum1 += dy[j] * gamma[channel];
            sum2 += dy[j] * gamma[channel] * normalized[j];
        }
    }

    // Reduce sums across the warp
    warpReduceLayernormSums(sum1, sum2);

    // Reduce sums across the block
    __shared__ float shared_sum1[32];
    __shared__ float shared_sum2[32];

    if (lane_id == 0)
    {
        shared_sum1[warp_id] = sum1;
        shared_sum2[warp_id] = sum2;
    }

    __syncthreads();

    // Final reduction by the first warp
    if (warp_id == 0)
    {
        sum1 = lane_id < warp_count ? shared_sum1[lane_id] : 0.0f;
        sum2 = lane_id < warp_count ? shared_sum2[lane_id] : 0.0f;

        warpReduceLayernormSums(sum1, sum2);

        if (lane_id == 0)
        {
            shared_sum1[0] = sum1;
            shared_sum2[0] = sum2;
        }
    }

    __syncthreads();

    // Broadcast the final sums to all threads in the block
    float final_sum1 = shared_sum1[0];
    float final_sum2 = shared_sum2[0];

    // Compute final gradients for x
    for (int i = thread_id; i < num_float4; i += blockDim.x)
    {
        float4 x_val = x4[i];
        float4 dy_val = grad_y4[i];
        float4 residual_val = grad_residual4 ? grad_residual4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float normalized[4] = {
            (x_val.x - mean) * inv_std,
            (x_val.y - mean) * inv_std,
            (x_val.z - mean) * inv_std,
            (x_val.w - mean) * inv_std,
        };
        float dy[4] = {dy_val.x, dy_val.y, dy_val.z, dy_val.w};
        float res[4] = {residual_val.x, residual_val.y, residual_val.z, residual_val.w};

        for (int j = 0; j < 4; j++)
        {
            int channel = i * 4 + j;

            grad_x[TENSOR_IDX_2D(row_idx, channel, d_model)] = inv_std * (dy[j] * gamma[channel] - final_sum1 / d_model - normalized[j] * final_sum2 / d_model) + res[j];
        }
    }
}

__inline__ __device__ void warpReduceLayernormGradients(
    float &grad_gamma,
    float &grad_beta)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        grad_gamma += __shfl_down_sync(0xffffffff, grad_gamma, offset);
        grad_beta += __shfl_down_sync(0xffffffff, grad_beta, offset);
    }
}

__global__ void layernorm_backward_weights_bias_f32_kernel(
    float *__restrict__ grad_gamma,
    float *__restrict__ grad_beta,
    const float *__restrict__ grad_y,
    const float *__restrict__ x,
    const float *__restrict__ means,
    const float *__restrict__ vars,
    int n,
    int d_model)
{
    const int thread_id = threadIdx.x;
    const int col = blockIdx.x;
    const int warp_count = blockDim.x / warpSize;
    const int warp_id = thread_id / warpSize;
    const int lane_id = thread_id % warpSize;

    if (col >= d_model)
    {
        return;
    }

    // Compute local gradients for gamma and beta
    float local_grad_gamma = 0.0f;
    float local_grad_beta = 0.0f;

    for (int row = thread_id; row < n; row += blockDim.x)
    {
        float mean = means[row];
        float inv_std = vars[row];

        float x_val = x[TENSOR_IDX_2D(row, col, d_model)];
        float dy_val = grad_y[TENSOR_IDX_2D(row, col, d_model)];

        float normalized = (x_val - mean) * inv_std;

        local_grad_gamma += dy_val * normalized;
        local_grad_beta += dy_val;
    }

    // Warp reduction for gradients
    warpReduceLayernormGradients(local_grad_gamma, local_grad_beta);

    // Reduce sums across the block
    __shared__ float shared_grad_gamma[32];
    __shared__ float shared_grad_beta[32];

    if (lane_id == 0)
    {
        shared_grad_gamma[warp_id] = local_grad_gamma;
        shared_grad_beta[warp_id] = local_grad_beta;
    }

    __syncthreads();

    // Final reduction by the first warp
    if (warp_id == 0)
    {
        local_grad_gamma = lane_id < warp_count ? shared_grad_gamma[lane_id] : 0.0f;
        local_grad_beta = lane_id < warp_count ? shared_grad_beta[lane_id] : 0.0f;

        warpReduceLayernormGradients(local_grad_gamma, local_grad_beta);

        if (lane_id == 0)
        {
            grad_gamma[col] = local_grad_gamma;
            grad_beta[col] = local_grad_beta;
        }
    }
}

void CUDABackend::device_layernorm_residual_fused_backward(float *grad_x, float *grad_gamma, float *grad_beta, const float *grad_y, const float *grad_residual, const float *x, const float *means, const float *vars, const float *gamma, int batch_size, int seq_len, int hidden_size)
{
    const int n = batch_size * seq_len;
    const int block_size = min(1024, hidden_size / 4);

    layernorm_residual_fused_backward_f32_v4_kernel<<<n, block_size>>>(
        grad_x,
        grad_gamma,
        grad_beta,
        grad_y,
        grad_residual,
        x,
        means,
        vars,
        gamma,
        n,
        hidden_size);

    CUDA_KERNEL_CHECK();

    layernorm_backward_weights_bias_f32_kernel<<<hidden_size, 256>>>(
        grad_gamma,
        grad_beta,
        grad_y,
        x,
        means,
        vars,
        n,
        hidden_size);

    CUDA_KERNEL_CHECK();
}