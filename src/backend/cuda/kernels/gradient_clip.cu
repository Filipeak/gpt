#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include "tensor_utils.h"

#define EPSILON 1e-6f

void CUDABackend::device_clip_grad_norm(float *g, int size, float max_norm)
{
    float norm;
    CUBLAS_CHECK(cublasSnrm2(cublas_handle_, size, g, 1, &norm)); // L2 norm of whole buffer
    CUDA_DEBUG_SYNC();

    if (norm > max_norm)
    {
        float scale = max_norm / (norm + EPSILON);
        CUBLAS_CHECK(cublasSscal(cublas_handle_, size, &scale, g, 1)); // g *= scale
        CUDA_DEBUG_SYNC();
    }
}