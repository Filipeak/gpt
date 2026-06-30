add_subdirectory(common)
add_subdirectory(cpu)
add_subdirectory(cuda)

add_library(gpt_backend INTERFACE)

target_link_libraries(gpt_backend
    INTERFACE
        backend_common
        backend_cpu
        backend_cuda
)

target_include_directories(gpt_backend
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)