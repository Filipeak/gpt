add_library(backend_cpu
    STATIC
        cpu_backend.cpp
)

target_link_libraries(backend_cpu
    PRIVATE
        backend_common
)

target_include_directories(backend_cpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)