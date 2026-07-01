#pragma once

#include "backend_interface.h"
#include <cstddef>

class DataLoader
{
public:
    DataLoader(IGPTBackend *backend, int batch_size, int seq_len);
    ~DataLoader();

    bool load_data(const char *data_path);

    void reset_epoch();
    bool next_batch();

    const int *inputs() const { return d_inputs_; }
    const int *labels() const { return d_labels_; }

    int current_batch() const { return current_sample_index_ / batch_size_; }
    int total_batches() const { return samples_count_ / batch_size_; }

private:
    IGPTBackend *backend_ = nullptr;
    int batch_size_ = 0;
    int seq_len_ = 0;

    int *data_ = nullptr;
    size_t data_size_ = 0;

    int *samples_indices_ = nullptr;
    size_t samples_count_ = 0;
    size_t current_sample_index_ = 0;

    int *tmp_inputs_ = nullptr;
    int *tmp_labels_ = nullptr;
    int *d_inputs_ = nullptr;
    int *d_labels_ = nullptr;
};