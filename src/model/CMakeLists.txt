# Library for the GPT model
add_library(gpt_model 
    STATIC
        gpt.cpp
)

target_link_libraries(gpt_model
    PUBLIC
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
        gpt_model
        gpt_backend
)