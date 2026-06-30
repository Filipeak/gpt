#include "data_loader.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

DataLoader::DataLoader(IGPTBackend *backend, const char *data_path, int batch_size, int seq_len) : backend_(backend), batch_size_(batch_size), seq_len_(seq_len), data_(nullptr), data_size_(0), samples_count_(0), samples_indices_(nullptr), current_sample_index_(0), tmp_inputs_(nullptr), tmp_labels_(nullptr), d_inputs_(nullptr), d_labels_(nullptr)
{
    load_data(data_path);

    if (data_size_ <= 2 * seq_len_)
    {
        printf("Error: Not enough data samples to create even a single batch. Data size: %zu, Sequence length: %d\n", data_size_, seq_len_);
        return;
    }

    samples_count_ = data_size_ / seq_len_ - 1; // -1 because we will add offset at the begining
    samples_indices_ = (int *)malloc(samples_count_ * sizeof(int));

    tmp_inputs_ = (int *)malloc(batch_size_ * seq_len_ * sizeof(int));
    tmp_labels_ = (int *)malloc(batch_size_ * seq_len_ * sizeof(int));

    backend_->device_malloc((void **)&d_inputs_, batch_size_ * seq_len_ * sizeof(int));
    backend_->device_malloc((void **)&d_labels_, batch_size_ * seq_len_ * sizeof(int));

    printf("Total training batches: %d\n", total_batches());
}

DataLoader::~DataLoader()
{
    if (data_)
    {
        free(data_);
        data_ = nullptr;
    }

    if (samples_indices_)
    {
        free(samples_indices_);
        samples_indices_ = nullptr;
    }

    if (tmp_inputs_)
    {
        free(tmp_inputs_);
        tmp_inputs_ = nullptr;
    }

    if (tmp_labels_)
    {
        free(tmp_labels_);
        tmp_labels_ = nullptr;
    }

    if (d_inputs_)
    {
        backend_->device_free(d_inputs_);
        d_inputs_ = nullptr;
    }

    if (d_labels_)
    {
        backend_->device_free(d_labels_);
        d_labels_ = nullptr;
    }
}

void DataLoader::reset_epoch()
{
    size_t base_offset = rand() % seq_len_; // Random offset to start from within the first sequence length

    for (size_t i = 0; i < samples_count_; ++i)
    {
        samples_indices_[i] = i * seq_len_ + base_offset;
    }

    // Shuffle the sample indices using Fisher-Yates shuffle
    for (size_t i = samples_count_ - 1; i > 0; i--)
    {
        size_t j = rand() % (i + 1);
        int temp = samples_indices_[i];
        samples_indices_[i] = samples_indices_[j];
        samples_indices_[j] = temp;
    }

    current_sample_index_ = 0;
}

bool DataLoader::next_batch()
{
    if (current_sample_index_ + batch_size_ - 1 >= samples_count_)
    {
        return false;
    }

    for (int i = 0; i < batch_size_; i++)
    {
        int sample_index = samples_indices_[current_sample_index_ + i];

        memcpy(&tmp_inputs_[i * seq_len_], &data_[sample_index], seq_len_ * sizeof(int));
        memcpy(&tmp_labels_[i * seq_len_], &data_[sample_index + 1], seq_len_ * sizeof(int));
    }

    // Copy the temporary arrays to the device memory
    backend_->device_memcpy_h2d(d_inputs_, tmp_inputs_, batch_size_ * seq_len_ * sizeof(int));
    backend_->device_memcpy_h2d(d_labels_, tmp_labels_, batch_size_ * seq_len_ * sizeof(int));

    current_sample_index_ += batch_size_;

    return true;
}

void DataLoader::load_data(const char *data_path)
{
    FILE *fp = fopen(data_path, "rb");

    if (!fp)
    {
        printf("Error: Could not open file %s for reading.\n", data_path);
        return;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    size_t count = size / sizeof(uint16_t);
    fseek(fp, 0, SEEK_SET);

    uint16_t *tmp_data = (uint16_t *)malloc(size);

    if (!tmp_data)
    {
        fclose(fp);
        printf("Error: Could not allocate memory for training data.\n");
        return;
    }

    size_t read_count = fread(tmp_data, sizeof(uint16_t), count, fp);
    fclose(fp);

    if (read_count != count)
    {
        free(tmp_data);
        printf("Error: Could not read all data from file %s.\n", data_path);
        return;
    }

    data_size_ = count;
    data_ = (int *)malloc(data_size_ * sizeof(int));

    if (!data_)
    {
        free(tmp_data);
        printf("Error: Could not allocate memory for training data. (final)\n");
        return;
    }

    for (size_t i = 0; i < data_size_; ++i)
    {
        data_[i] = (int)tmp_data[i];
    }

    free(tmp_data);
    printf("Loaded %zu training samples from %s\n", data_size_, data_path);
}