cmake_minimum_required(VERSION 3.25)

project(gpt LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GPT_DEBUG "Sync and check for CUDA errors after every op, naming the op that failed (slow)" OFF)

find_package(CUDAToolkit REQUIRED)

enable_testing()

add_subdirectory(src)

# Helper targets, none of them part of the default build
include(cmake/python.cmake) # data, grad_goldens
include(cmake/tools.cmake)  # sanitize, nsys
