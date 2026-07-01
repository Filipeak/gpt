add_executable(gpt_train
    train.cpp
    data_loader.cpp
    lr_scheduling.cpp
)

target_link_libraries(gpt_train
    PRIVATE
        gpt_core
        gpt_model
)