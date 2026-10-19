add_library(backend_cuda
    STATIC
        core/cuda_backend.cu
        core/cuda_memory.cu
        kernels/adamw.cu
        kernels/attention.cu
        kernels/cross_entropy.cu
        kernels/embedding.cu
        kernels/gradient_clip.cu
        kernels/layernorm.cu
        kernels/linear.cu
        kernels/softmax.cu
)

target_include_directories(backend_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(backend_cuda
    PRIVATE
        backend_common
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        CUDA::nvtx3
)

# Source line info for compute-sanitizer and Nsight Compute reports, it does not slow the kernels down.
target_compile_options(backend_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

if(GPT_DEBUG)
    target_compile_definitions(backend_cuda PRIVATE GPT_DEBUG)
endif()

set_target_properties(backend_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(backend_cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)