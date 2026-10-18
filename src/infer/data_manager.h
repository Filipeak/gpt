#pragma once

#include "backend_interface.h"
#include <cstddef>
#include <vector>

class DataManager
{
public:
    DataManager(IGPTBackend *backend);
    ~DataManager();

    void push_token(int token);
    bool load_data(const char *data_path, int additional_tokens);
    bool load_tokens(const std::vector<int> &tokens, int additional_tokens);
    bool save_data(const char *output_path);

    const int *device_data();
    const int *host_data() const { return data_; }
    size_t current_size() const { return current_index_; }

private:
    IGPTBackend *backend_ = nullptr;
    int *data_ = nullptr;
    int *d_data_ = nullptr;
    size_t data_size_ = 0;
    size_t current_index_ = 0;
};