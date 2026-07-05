#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"

CUDABackend::CUDABackend(unsigned long seed)
{
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH));

    CURAND_CHECK(curandCreateGenerator(&curand_handle_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed((curandGenerator_t)curand_handle_, seed));

    CUDA_CHECK(cudaMalloc(&clip_norm_device_, sizeof(float)));
}

CUDABackend::~CUDABackend()
{
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));

    CURAND_CHECK(curandDestroyGenerator(curand_handle_));

    CUDA_CHECK(cudaFree(clip_norm_device_));
}