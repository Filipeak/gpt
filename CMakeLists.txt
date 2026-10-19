cmake_minimum_required(VERSION 3.25)

project(gpt LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

# Profiling: cmake --build build --config Release --target nsys
# Records a few training steps and prints GPU time per NVTX range and per kernel.
file(GLOB NSYS_HINTS "C:/Program Files/NVIDIA Corporation/Nsight Systems */target-windows-x64")
find_program(NSYS_EXECUTABLE nsys HINTS ${NSYS_HINTS})

if(NSYS_EXECUTABLE)
    set(PROFILES_DIR ${CMAKE_BINARY_DIR}/profiles)

    add_custom_target(nsys
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROFILES_DIR}
        COMMAND ${NSYS_EXECUTABLE} profile --trace=cuda,nvtx --cuda-event-trace=false --sample=none --cpuctxsw=none --force-overwrite=true
                --output=${PROFILES_DIR}/train
                $<TARGET_FILE:gpt_train> --epochs 1 --max-steps 3 --batch-size 8 --batch-accum-steps 4
                --data ${CMAKE_SOURCE_DIR}/data/tiny_shakespeare.bin
        COMMAND ${NSYS_EXECUTABLE} stats --force-export=true --report nvtx_gpu_proj_sum --report cuda_gpu_kern_sum
                ${PROFILES_DIR}/train.nsys-rep
        DEPENDS gpt_train
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
        COMMENT "Profiling 3 training steps with Nsight Systems")
else()
    message(STATUS "nsys not found, the nsys target is unavailable")
endif()
