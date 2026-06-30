#pragma once

#include "backend_interface.h"
#include <cstddef>

class DataLoader
{
public:
    DataLoader(IGPTBackend *backend, const char *data_path, int batch_size, int seq_len);
    ~DataLoader();

    void reset_epoch();
    bool next_batch();

    const int *inputs() const { return d_inputs_; }
    const int *labels() const { return d_labels_; }

    int current_batch() const { return current_sample_index_ / batch_size_; }
    int total_batches() const { return samples_count_ / batch_size_; }

private:
    IGPTBackend *backend_;
    int batch_size_;
    int seq_len_;

    int *data_;
    size_t data_size_;

    int *samples_indices_;
    size_t samples_count_;
    size_t current_sample_index_;

    int *tmp_inputs_;
    int *tmp_labels_;
    int *d_inputs_;
    int *d_labels_;

    void load_data(const char *data_path);
};