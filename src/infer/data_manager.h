#pragma once

#include "backend_interface.h"
#include <cstddef>

class DataManager
{
public:
    DataManager(IGPTBackend *backend, const char *input_path, int tokens_to_generate);
    ~DataManager();

    void push_token(int token);
    void save_data(const char *output_path);

    const int *device_data();
    size_t current_size() const { return current_index_; }

private:
    IGPTBackend *backend_;
    int *data_;
    int *d_data_;
    size_t data_size_;
    size_t current_index_;

    void load_data(const char *data_path, int additional_tokens);
};