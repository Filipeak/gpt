#include "cuda_backend.h"
#include "utils/cuda_utils.cuh"
#include <nvtx3/nvToolsExt.h>

CUDABackend::CUDABackend(unsigned long seed)
{
#ifdef GPT_DEBUG
    fprintf(stderr, "[warn] GPT_DEBUG build: syncing and checking for CUDA errors after every op, timings are not representative.\n");
#endif

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

// NVTX calls are near free when no profiler is attached.
void CUDABackend::profile_range_push(const char *name)
{
#ifdef GPT_DEBUG
    range_stack_.push_back(name);
#endif

    nvtxRangePushA(name);
}

void CUDABackend::profile_range_pop()
{
#ifdef GPT_DEBUG
    // Every op runs inside a range, so syncing here reports an error against the op that caused it,
    // e.g. "backward > layer_3 > layernorm_bwd", instead of at some later cudaMemcpy.
    cudaError_t result = cudaDeviceSynchronize();

    if (result == cudaSuccess)
    {
        result = cudaGetLastError();
    }

    if (result != cudaSuccess)
    {
        std::string path;

        for (const std::string &name : range_stack_)
        {
            path += path.empty() ? name : " > " + name;
        }

        fprintf(stderr, "[error] CUDA error in %s: %s\n", path.c_str(), cudaGetErrorString(result));
        abort();
    }

    range_stack_.pop_back();
#endif

    nvtxRangePop();
}