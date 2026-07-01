add_executable(gpt_infer
    infer.cpp
    data_manager.cpp
    token_sampler.cpp
)

target_link_libraries(gpt_infer
    PRIVATE
        gpt_core
        gpt_model
)