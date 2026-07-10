#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"
#include <float.h>

template <const int Br, const int Bc>
__global__ void flash_attention_2_forward_fp32_kernel(
    float *__restrict__ y,
    float *__restrict__ L,
    const float *__restrict__ qkv,
    int seq_len,
    int hidden_size,
    int num_heads)
{
    // Locate current thread
    const int block = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;

    const int thread_id = threadIdx.x; // We use flat block of threads
    const int lane_id = thread_id % warpSize;
    const int warp_id = thread_id / warpSize;
    const int warp_count = blockDim.x / warpSize;

    // Calculate helper sizes
    const int d_head = hidden_size / num_heads;
    const float inv_sqrt_d_head = rsqrtf((float)d_head);
    const int part_size_row = Br / warp_count;

    // Prepare shared memory for Q, K, V, O
    extern __shared__ float shared_mem[];

    __shared__ float rowmax_shared[Br];
    __shared__ float rowsumexp_shared[Br];

    const int br_size = Br * d_head;
    const int bc_size = Bc * d_head;

    // TODO: Use vectorized loads for Q, K, V, O to improve memory throughput
    float *q_shared = shared_mem;
    float *k_shared = q_shared + br_size;
    float *v_shared = k_shared + bc_size + d_head; // +d_head for padding to avoid bank conflicts (Bc + 1)
    float *o_shared = v_shared + bc_size;
    float *rowcache_shared = o_shared + br_size;

    // Load Q, Reset O, Reset rowmax and rowsumexp
    const int qkv_q_offset = batch * seq_len * hidden_size * 3 + head * d_head;

    for (int i = thread_id; i < br_size; i += blockDim.x)
    {
        const int global_idx = block * br_size + i;
        const int seq_idx = global_idx / d_head;

        // Check for out-of-bounds access
        if (seq_idx >= seq_len)
        {
            q_shared[i] = 0.0f;
            o_shared[i] = 0.0f;
        }
        else
        {
            const int head_idx = global_idx % d_head;
            const int qkv_q_idx = qkv_q_offset + seq_idx * hidden_size * 3 + head_idx;

            q_shared[i] = qkv[qkv_q_idx];
            o_shared[i] = 0.0f;

            // Initialize rowmax and rowsumexp
            if (i % d_head == 0) // Only one thread per row initializes
            {
                rowmax_shared[i / d_head] = -FLT_MAX;
                rowsumexp_shared[i / d_head] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Inner loop
    for (int current_col_block = 0; current_col_block * Bc < Br * (block + 1); current_col_block++) // We do lower left triangle
    {
        // Load K and V
        // TODO: Use async copy (double buffering)
        for (int j = thread_id; j < bc_size; j += blockDim.x)
        {
            const int global_idx = current_col_block * bc_size + j;
            const int seq_idx = global_idx / d_head;

            // Transpose K for coalesced access
            const int local_row = j / d_head;
            const int local_col = j % d_head;
            const int local_trans_idx = local_col * (Bc + 1) + local_row; // stride of (Bc + 1) to avoid bank conflicts

            // Check for out-of-bounds access
            if (seq_idx >= seq_len)
            {
                k_shared[local_trans_idx] = 0.0f;
                v_shared[j] = 0.0f;
            }
            else
            {
                const int head_idx = global_idx % d_head;
                const int qkv_k_idx = qkv_q_offset + hidden_size + seq_idx * hidden_size * 3 + head_idx;

                k_shared[local_trans_idx] = qkv[qkv_k_idx];
                v_shared[j] = qkv[qkv_k_idx + hidden_size];
            }
        }

        __syncthreads();

        // Compute attention scores online and update O
        for (int local_row = 0; local_row < part_size_row; local_row++)
        {
            const int block_row = warp_id * part_size_row + local_row;
            const int global_row = block * Br + block_row;

            if (global_row >= seq_len)
            {
                continue; // Skip out-of-bounds rows
            }

            float local_max = -FLT_MAX;

            for (int block_col = lane_id; block_col < Bc; block_col += warpSize)
            {
                const int global_col = current_col_block * Bc + block_col;

                float x = 0.0f;

                // Compute dot product only for valid positions (lower triangle)
                if (global_col <= global_row)
                {
                    // Dot product Q*K^T
                    // TODO: use tensor cores for this (fp16 required)
                    for (int k = 0; k < d_head; k++)
                    {
                        x += q_shared[block_row * d_head + k] * k_shared[k * (Bc + 1) + block_col];
                    }

                    // Scale
                    x *= inv_sqrt_d_head;
                }
                else
                {
                    x = -FLT_MAX; // Mask future tokens
                }

                // Update cache and local max
                rowcache_shared[warp_id * Bc + block_col] = x;
                local_max = fmaxf(local_max, x);
            }

// Reduce local_max across the warp
#pragma unroll
            for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            {
                local_max = max(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
            }

            // Check new row max and compute correction factor
            float old_rowmax = rowmax_shared[block_row];
            float new_rowmax = fmaxf(local_max, old_rowmax);
            float correction = __expf(old_rowmax - new_rowmax);

            // Compute exp(x - new_rowmax) and accumulate local sumexp
            float local_sumexp = 0.0f;

            for (int block_col = lane_id; block_col < Bc; block_col += warpSize)
            {
                const int idx = warp_id * Bc + block_col;
                float val = __expf(rowcache_shared[idx] - new_rowmax);
                local_sumexp += val;
                rowcache_shared[idx] = val; // Store back for later use
            }

            __syncwarp();

// Reduce local_sumexp across the warp
#pragma unroll
            for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            {
                local_sumexp += __shfl_xor_sync(0xffffffff, local_sumexp, offset);
            }

            // Update rowsumexp with correction factor
            float old_rowsumexp = rowsumexp_shared[block_row];
            float new_rowsumexp = old_rowsumexp * correction + local_sumexp;

            // Finally accumulate the output O
            for (int d = lane_id; d < d_head; d += warpSize)
            {
                float acc = o_shared[block_row * d_head + d] * correction;

                // Compute dot product of rowcache and V
                for (int block_col = 0; block_col < Bc; block_col++)
                {
                    acc += rowcache_shared[warp_id * Bc + block_col] * v_shared[block_col * d_head + d];
                }

                o_shared[block_row * d_head + d] = acc;
            }

            // Update rowmax and rowsumexp in shared memory
            if (lane_id == 0)
            {
                rowmax_shared[block_row] = new_rowmax;
                rowsumexp_shared[block_row] = new_rowsumexp;
            }

            __syncwarp();
        }

        __syncthreads();
    }

    // Write O back to global memory
    for (int i = thread_id; i < br_size; i += blockDim.x)
    {
        const int global_idx = block * br_size + i;
        const int seq_idx = global_idx / d_head;

        // Out-of-bounds check
        if (seq_idx < seq_len)
        {
            const int head_idx = global_idx % d_head;
            const int y_idx = batch * seq_len * hidden_size + head * d_head + seq_idx * hidden_size + head_idx;
            const int row_idx = i / d_head;

            y[y_idx] = o_shared[i] / rowsumexp_shared[row_idx]; // Set and normalize by sumexp
        }
    }

    // Write L
    for (int row = thread_id; row < Br; row += blockDim.x)
    {
        const int global_row = block * Br + row;

        // Out-of-bounds check
        if (global_row < seq_len)
        {
            const int l_idx = batch * seq_len * num_heads + global_row * num_heads + head;

            L[l_idx] = rowmax_shared[row] + logf(rowsumexp_shared[row]);
        }
    }
}

void CUDABackend::device_flash_attention_forward(float *y, float *logsumexp, const float *qkv, int batch_size, int seq_len, int hidden_size, int num_heads)
{
    const int num_warps = 4;
    const int Br = 16; // must be divisible by num_warps
    const int Bc = 32; // must be divisible by 32

    const dim3 block_size(num_warps * 32);
    const dim3 grid_size((seq_len + Br - 1) / Br, num_heads, batch_size);
    const int shared_mem_size = ((Br + Bc + Bc + 1 + Br) * (hidden_size / num_heads) + Bc * num_warps) * sizeof(float);

    flash_attention_2_forward_fp32_kernel<Br, Bc><<<grid_size, block_size, shared_mem_size>>>(
        y,
        logsumexp,
        qkv,
        seq_len,
        hidden_size,
        num_heads);

    CUDA_KERNEL_CHECK();
}

__global__ void flash_attention_precompute_rowsums_kernel(
    float *__restrict__ D,
    const float *__restrict__ y,
    const float *__restrict__ grad_y,
    int hidden_size,
    int num_heads)
{
    const int global_row = blockIdx.x;
    const int current_head = blockIdx.y;
    const int thread_id = threadIdx.x;
    const int lane_id = thread_id % warpSize;

    const int d_head = hidden_size / num_heads;
    const int idx = global_row * hidden_size + current_head * d_head;

    // Compute local sum for the current thread
    float local_sum = 0.0f;

    for (int i = thread_id; i < d_head; i += blockDim.x)
    {
        local_sum += grad_y[idx + i] * y[idx + i];
    }

// Reduce the sum across the warp
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Assuming single warp per block, write the result to global memory
    if (lane_id == 0)
    {
        D[global_row * num_heads + current_head] = local_sum;
    }
}

template <const int Br, const int Bc>
__global__ void flash_attention_2_backward_fp32_kernel(
    float *__restrict__ grad_x,
    const float *__restrict__ grad_y,
    const float *__restrict__ qkv,
    const float *__restrict__ L,
    const float *__restrict__ D,
    int seq_len,
    int hidden_size,
    int num_heads)
{
    // Locate current thread
    const int block = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;

    const int thread_id = threadIdx.x; // We use flat block of threads
    const int lane_id = thread_id % warpSize;
    const int warp_id = thread_id / warpSize;
    const int warp_count = blockDim.x / warpSize;

    // Calculate helper sizes
    const int d_head = hidden_size / num_heads;
    const float inv_sqrt_d_head = rsqrtf((float)d_head);
    const int part_size_col = Bc / warp_count;

    // Prepare shared memory
    extern __shared__ float shared_mem[];

    const int br_size = Br * d_head;
    const int bc_size = Bc * d_head;

    // TODO: Use vectorized loads
    float *q_shared = shared_mem;
    float *k_shared = q_shared + br_size + d_head; // +d_head for padding to avoid bank conflicts
    float *v_shared = k_shared + bc_size;
    float *dO_shared = v_shared + bc_size;
    float *dK_shared = dO_shared + br_size + d_head; // +d_head for padding to avoid bank conflicts
    float *dV_shared = dK_shared + bc_size;
    float *L_shared = dV_shared + bc_size;
    float *D_shared = L_shared + Br;
    float *p_shared = D_shared + Br;
    float *dS_shared = p_shared + Br * warp_count;

    // Load K and V, initialize dK and dV
    for (int i = thread_id; i < bc_size; i += blockDim.x)
    {
        const int global_idx = block * bc_size + i;
        const int seq_idx = global_idx / d_head;

        if (seq_idx < seq_len)
        {
            const int head_idx = global_idx % d_head;
            const int qkv_k_idx = batch * seq_len * hidden_size * 3 + head * d_head + hidden_size + seq_idx * hidden_size * 3 + head_idx;

            k_shared[i] = qkv[qkv_k_idx];
            v_shared[i] = qkv[qkv_k_idx + hidden_size];
        }
        else
        {
            k_shared[i] = 0.0f;
            v_shared[i] = 0.0f;
        }

        dK_shared[i] = 0.0f;
        dV_shared[i] = 0.0f;
    }

    __syncthreads();

    // Inner loop
    const int start_idx = (seq_len + Br - 1) / Br - 1;

    for (int current_row_block = start_idx; block * Bc < Br * (current_row_block + 1); current_row_block--) // Start from the bottom, do lower left triangle
    {
        // Load Q, O, dO, L, D
        // TODO: Use async copy (double buffering)
        for (int j = thread_id; j < br_size; j += blockDim.x)
        {
            const int global_idx = current_row_block * br_size + j;
            const int seq_idx = global_idx / d_head;

            // Transpose for coalesced access
            const int local_row = j / d_head;
            const int local_col = j % d_head;
            const int local_trans_idx = local_col * (Br + 1) + local_row;

            if (seq_idx < seq_len)
            {
                const int head_idx = global_idx % d_head;
                const int qkv_q_idx = batch * seq_len * hidden_size * 3 + head * d_head + seq_idx * hidden_size * 3 + head_idx;
                const int dO_idx = batch * seq_len * hidden_size + head * d_head + seq_idx * hidden_size + head_idx;

                q_shared[local_trans_idx] = qkv[qkv_q_idx];
                dO_shared[local_trans_idx] = grad_y[dO_idx];

                if (j % d_head == 0) // Only one thread per row writes L and D
                {
                    const int idx = batch * seq_len * num_heads + seq_idx * num_heads + head;

                    L_shared[j / d_head] = L[idx];
                    D_shared[j / d_head] = D[idx];
                }
            }
            else
            {
                q_shared[local_trans_idx] = 0.0f;
                dO_shared[local_trans_idx] = 0.0f;
                L_shared[j / d_head] = 0.0f;
                D_shared[j / d_head] = 0.0f;
            }
        }

        // Reset dS
        for (int i = thread_id; i < Bc * Br; i += blockDim.x)
        {
            dS_shared[i] = 0.0f;
        }

        __syncthreads();

        // Compute intermediate gradients
        for (int local_col = 0; local_col < part_size_col; local_col++)
        {
            const int block_col = warp_id * part_size_col + local_col;
            const int global_col = block * Bc + block_col;

            if (global_col >= seq_len)
            {
                continue; // Skip out-of-bounds columns
            }

            for (int block_row = lane_id; block_row < Br; block_row += warpSize)
            {
                const int global_row = current_row_block * Br + block_row;

                float x = 0.0f;

                if (global_col <= global_row) // Only compute for lower triangle
                {
                    // Dot product Q*K^T
                    // TODO: use tensor cores for this (fp16 required)
                    for (int k = 0; k < d_head; k++)
                    {
                        x += q_shared[k * (Br + 1) + block_row] * k_shared[block_col * d_head + k];
                    }

                    // Scale
                    x *= inv_sqrt_d_head;

                    // Compute intermediate gradients and values
                    float dP = 0.0f;

                    for (int k = 0; k < d_head; k++)
                    {
                        dP += dO_shared[k * (Br + 1) + block_row] * v_shared[block_col * d_head + k];
                    }

                    float p = __expf(x - L_shared[block_row]);
                    float dS = p * (dP - D_shared[block_row]) * inv_sqrt_d_head;

                    // Save results to shared memory for later use
                    p_shared[warp_id * Br + block_row] = p;
                    dS_shared[block_col * Br + block_row] = dS;
                }
                else
                {
                    p_shared[warp_id * Br + block_row] = 0.0f;
                    dS_shared[block_col * Br + block_row] = 0.0f;
                }
            }

            __syncwarp();

            // Compute dK, dV
            for (int d = lane_id; d < d_head; d += warpSize)
            {
                float dV_acc = 0.0f;
                float dK_acc = 0.0f;

                for (int block_row = 0; block_row < Br; block_row++)
                {
                    dV_acc += p_shared[warp_id * Br + block_row] * dO_shared[d * (Br + 1) + block_row];
                    dK_acc += dS_shared[block_col * Br + block_row] * q_shared[d * (Br + 1) + block_row];
                }

                dV_shared[block_col * d_head + d] += dV_acc;
                dK_shared[block_col * d_head + d] += dK_acc;
            }

            __syncwarp();
        }

        __syncthreads();

        // Compute dQ (one atomicAdd per tile)
        for (int block_row = warp_id; block_row < Br; block_row += warp_count)
        {
            const int seq_idx = current_row_block * Br + block_row;

            if (seq_idx < seq_len)
            {
                for (int d = lane_id; d < d_head; d += warpSize)
                {
                    float dQ_acc = 0.0f;

                    for (int c = 0; c < Bc; c++)
                    {
                        dQ_acc += dS_shared[c * Br + block_row] * k_shared[c * d_head + d];
                    }

                    const int dQ_idx = batch * seq_len * hidden_size * 3 + head * d_head + seq_idx * hidden_size * 3 + d;

                    atomicAdd(&grad_x[dQ_idx], dQ_acc);
                }
            }
        }

        __syncthreads();
    }

    // Copy dK and dV back to global memory
    for (int i = thread_id; i < bc_size; i += blockDim.x)
    {
        const int global_idx = block * bc_size + i;
        const int seq_idx = global_idx / d_head;

        if (seq_idx < seq_len)
        {
            const int head_idx = global_idx % d_head;
            const int dK_idx = batch * seq_len * hidden_size * 3 + head * d_head + hidden_size + seq_idx * hidden_size * 3 + head_idx;

            grad_x[dK_idx] += dK_shared[i];
            grad_x[dK_idx + hidden_size] += dV_shared[i];
        }
    }
}

void CUDABackend::device_flash_attention_backward(float *grad_x, float *D_cache, const float *logsumexp, const float *grad_y, const float *y, const float *qkv, int batch_size, int seq_len, int hidden_size, int num_heads)
{
    CUDA_CHECK(cudaMemset(grad_x, 0, batch_size * seq_len * hidden_size * 3 * sizeof(float)));

    flash_attention_precompute_rowsums_kernel<<<dim3(batch_size * seq_len, num_heads), 32>>>(
        D_cache,
        y,
        grad_y,
        hidden_size,
        num_heads);

    CUDA_KERNEL_CHECK();

    const int num_warps = 4;
    const int Br = 32; // must be divisible by 32
    const int Bc = 16; // must be divisible by num_warps

    const dim3 block_size(num_warps * 32);
    const dim3 grid_size((seq_len + Bc - 1) / Bc, num_heads, batch_size);
    const int shared_mem_size = ((Br + 1 + Bc + Bc + Br + 1 + Bc + Bc) * (hidden_size / num_heads) + Br + Br + Br * num_warps + Bc * Br) * sizeof(float);

    flash_attention_2_backward_fp32_kernel<Br, Bc><<<grid_size, block_size, shared_mem_size>>>(
        grad_x,
        grad_y,
        qkv,
        logsumexp,
        D_cache,
        seq_len,
        hidden_size,
        num_heads);

    CUDA_KERNEL_CHECK();
}