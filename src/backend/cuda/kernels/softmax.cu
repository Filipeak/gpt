#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"
#include <float.h>
#include <math.h>

__inline__ __device__ void warpReduceOnlineSoftmax(
    float &max_val,
    float &sum_val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        float other_sum = __shfl_down_sync(0xffffffff, sum_val, offset);
        float new_max = max(max_val, other_max);

        sum_val = sum_val * expf(max_val - new_max) + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

__global__ void softmax_f32_v4_kernel(
    float *__restrict__ y,
    const float *__restrict__ x,
    int n,
    int vocab_size,
    int vocab_size_padded)
{
    const int thread_id = threadIdx.x;
    const int row_idx = blockIdx.x;
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;

    if (row_idx >= n)
    {
        return;
    }

    // Use float4 to process 4 elements at a time for better memory coalescing and performance (assuming vocab_size is a multiple of 4 and memory alignment allows it)
    float4 *x4 = (float4 *)(x + row_idx * vocab_size_padded);
    float4 *y4 = (float4 *)(y + row_idx * vocab_size_padded);
    int num_float4 = vocab_size_padded / 4;

    // Compute max and sum in single pass for current lane
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;

    for (int i = thread_id; i < num_float4; i += blockDim.x)
    {
        float4 val4 = x4[i];
        float vals[4] = {val4.x, val4.y, val4.z, val4.w};

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float v = i * 4 + j < vocab_size ? vals[j] : -FLT_MAX; // Ignore padded values
            float new_max = max(max_val, v);

            sum_val = sum_val * expf(max_val - new_max) + expf(v - new_max);
            max_val = new_max;
        }
    }

    // Warp reduction
    warpReduceOnlineSoftmax(max_val, sum_val);

    // Block reduction (block size is 1024 threads, so 32 warps)
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    if (lane_id == 0)
    {
        shared_max[warp_id] = max_val;
        shared_sum[warp_id] = sum_val;
    }

    __syncthreads();

    // Final reduction by the first warp
    if (warp_id == 0)
    {
        // Each lane from the first warp reads the max and sum from shared memory
        max_val = (lane_id < 32) ? shared_max[lane_id] : -FLT_MAX;
        sum_val = (lane_id < 32) ? shared_sum[lane_id] : 0.0f;

        warpReduceOnlineSoftmax(max_val, sum_val);

        if (lane_id == 0)
        {
            shared_max[0] = max_val;
            shared_sum[0] = sum_val;
        }
    }

    __syncthreads();

    // // Broadcast the final max and sum to all threads
    float global_max = shared_max[0];
    float global_sum = shared_sum[0];

    // Compute softmax for current lane (second pass)
    for (int i = thread_id; i < num_float4; i += blockDim.x)
    {
        float4 val4 = x4[i];

        // Compute softmax for each of the 4 values in val4 (ignore padded values)
        y4[i].x = i * 4 + 0 < vocab_size ? expf(val4.x - global_max) / global_sum : 0.0f;
        y4[i].y = i * 4 + 1 < vocab_size ? expf(val4.y - global_max) / global_sum : 0.0f;
        y4[i].z = i * 4 + 2 < vocab_size ? expf(val4.z - global_max) / global_sum : 0.0f;
        y4[i].w = i * 4 + 3 < vocab_size ? expf(val4.w - global_max) / global_sum : 0.0f;
    }
}

void CUDABackend::device_softmax_forward(float *y, const float *x, int batch_size, int seq_len, int vocab_size, int vocab_size_padded)
{
    const int n = batch_size * seq_len;
    const int block_size = 1024;
    const int grid_size = n;

    softmax_f32_v4_kernel<<<grid_size, block_size>>>(
        y,
        x,
        n,
        vocab_size,
        vocab_size_padded);

    CUDA_KERNEL_CHECK();
}