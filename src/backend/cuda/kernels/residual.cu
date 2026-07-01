#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

void CUDABackend::device_residual_forward(float *current, const float *residual, int batch_size, int seq_len, int hidden_size)
{
    const float alpha = 1.0f;
    const float beta = 1.0f;

    const int rows = batch_size * seq_len;
    const int cols = hidden_size;

    CUBLAS_CHECK(cublasSgeam(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cols, rows,
        &alpha,
        current, cols,
        &beta,
        residual, cols,
        current, cols));

    CUDA_KERNEL_CHECK();
}

void CUDABackend::device_residual_backward(float *grad_current, const float *grad_residual, int batch_size, int seq_len, int hidden_size)
{
    device_residual_forward(grad_current, grad_residual, batch_size, seq_len, hidden_size);
}