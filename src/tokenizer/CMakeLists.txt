# GPT-2 BPE tokenizer
add_library(gpt_tokenizer
    STATIC
        tokenizer.cpp
        unicode.cpp
)

target_link_libraries(gpt_tokenizer
    PUBLIC
        gpt_core
)

target_include_directories(gpt_tokenizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Testing executable for the tokenizer
add_executable(tokenizer_test
    tokenizer_test.cpp
)

target_link_libraries(tokenizer_test
    PRIVATE
        gpt_core
        gpt_tokenizer
)
