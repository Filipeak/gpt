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
    int d_model)
{
    const int thread_id = threadIdx.x;
    const int row_idx = blockIdx.x;
    const int warp_count = blockDim.x / warpSize;
    const int warp_id = thread_id / warpSize;
    const int lane_id = thread_id % warpSize;

    // Use float4 to process 4 elements at a time for better memory coalescing and performance (assuming d_model is a multiple of 4 and memory alignment allows it)
    float4 *x4 = (float4 *)(x + row_idx * d_model);
    float4 *y4 = (float4 *)(y + row_idx * d_model);
    const float4 *residual4 = residual ? (const float4 *)(residual + row_idx * d_model) : nullptr;
    const float4 *weight4 = (const float4 *)weight;
    const float4 *bias4 = (const float4 *)bias;
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
        float4 weight_val4 = weight4[i];
        float4 bias_val4 = bias4[i];
        float4 result;

        result.x = (val4.x - shared_mean) * shared_inv_std * weight_val4.x + bias_val4.x;
        result.y = (val4.y - shared_mean) * shared_inv_std * weight_val4.y + bias_val4.y;
        result.z = (val4.z - shared_mean) * shared_inv_std * weight_val4.z + bias_val4.z;
        result.w = (val4.w - shared_mean) * shared_inv_std * weight_val4.w + bias_val4.w;

        y4[i] = result;
    }
}

void CUDABackend::device_layernorm_residual_fused_forward(float *y, float *means, float *vars, float *x, const float *residual, const float *gamma, const float *beta, int batch_size, int seq_len, int hidden_size)
{
    CUDA_ASSERT(hidden_size % 4 == 0); // Ensure hidden_size is a multiple of 4 for float4 processing

    const int n = batch_size * seq_len;
    const int block_size = min(1024, ((hidden_size / 4 + 31) / 32) * 32); // Round up to the nearest multiple of 32 for warp alignment

    layernorm_residual_fused_forward_f32_v4_kernel<<<n, block_size>>>(
        y,
        means,
        vars,
        x,
        residual,
        gamma,
        beta,
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
    int d_model)
{
    const int thread_id = threadIdx.x;
    const int row_idx = blockIdx.x;
    const int warp_count = blockDim.x / warpSize;
    const int warp_id = thread_id / warpSize;
    const int lane_id = thread_id % warpSize;

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
    float final_sum1_div = shared_sum1[0] / d_model;
    float final_sum2_div = shared_sum2[0] / d_model;

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

            grad_x[TENSOR_IDX_2D(row_idx, channel, d_model)] = inv_std * (dy[j] * gamma[channel] - final_sum1_div - normalized[j] * final_sum2_div) + res[j];
        }
    }
}

// TODO: Use float4 for vectorized access and better memory coalescing
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
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Compute local gradients for gamma and beta
    float local_grad_gamma = 0.0f;
    float local_grad_beta = 0.0f;

    if (col < d_model)
    {
        for (int row = threadIdx.y; row < n; row += blockDim.y)
        {
            float mean = means[row];
            float inv_std = vars[row];

            float x_val = x[TENSOR_IDX_2D(row, col, d_model)];
            float dy_val = grad_y[TENSOR_IDX_2D(row, col, d_model)];

            float normalized = (x_val - mean) * inv_std;

            local_grad_gamma += dy_val * normalized;
            local_grad_beta += dy_val;
        }
    }

    // Use shared memory to accumulate gradients across threads in the block
    extern __shared__ float smem[];
    float *shared_grad_gamma = smem;
    float *shared_grad_beta = smem + blockDim.x * blockDim.y;

    shared_grad_gamma[tid] = local_grad_gamma;
    shared_grad_beta[tid] = local_grad_beta;

    __syncthreads();

    // Perform reduction within the block to accumulate gradients for gamma and beta (sum across columns)
    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.y < stride)
        {
            shared_grad_gamma[tid] += shared_grad_gamma[tid + stride * blockDim.x];
            shared_grad_beta[tid] += shared_grad_beta[tid + stride * blockDim.x];
        }

        __syncthreads();
    }

    // Finally write the accumulated gradients for gamma and beta to global memory
    if (threadIdx.y == 0 && col < d_model)
    {
        grad_gamma[col] += shared_grad_gamma[threadIdx.x];
        grad_beta[col] += shared_grad_beta[threadIdx.x];
    }
}

void CUDABackend::device_layernorm_residual_fused_backward(float *grad_x, float *grad_gamma, float *grad_beta, const float *grad_y, const float *grad_residual, const float *x, const float *means, const float *vars, const float *gamma, int batch_size, int seq_len, int hidden_size)
{
    CUDA_ASSERT(hidden_size % 4 == 0); // Ensure hidden_size is a multiple of 4 for float4 processing

    const int n = batch_size * seq_len;
    const int block_size = min(1024, ((hidden_size / 4 + 31) / 32) * 32); // Round up to the nearest multiple of 32 for warp alignment

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
        hidden_size);

    CUDA_KERNEL_CHECK();

    const dim3 wb_block(32, 32);
    const dim3 wb_grid((hidden_size + wb_block.x - 1) / wb_block.x);
    const size_t wb_smem = 2 * wb_block.x * wb_block.y * sizeof(float);

    layernorm_backward_weights_bias_f32_kernel<<<wb_grid, wb_block, wb_smem>>>(
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