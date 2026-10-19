# Library for the GPT model
add_library(gpt_model
    STATIC
        gpt.cpp
)

target_link_libraries(gpt_model
    PUBLIC
        gpt_core
        gpt_backend
)

target_include_directories(gpt_model
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Testing executable for the GPT model
add_executable(gpt_test
    gpt_test.cpp
)

target_link_libraries(gpt_test
    PRIVATE
        gpt_core
        gpt_model
        gpt_backend
)

# Gradient check against the CPU backend and PyTorch: ctest --test-dir build -C Release
# It reads the goldens from ./data, so it runs from the repo root.
add_test(NAME gpt_test COMMAND gpt_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})