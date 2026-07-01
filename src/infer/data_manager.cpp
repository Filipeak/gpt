#include "data_manager.h"
#include "logger.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

DataManager::DataManager(IGPTBackend *backend, const char *input_path, int tokens_to_generate) : backend_(backend), data_(nullptr), d_data_(nullptr), data_size_(0), current_index_(0)
{
    load_data(input_path, tokens_to_generate);

    backend_->device_malloc((void **)&d_data_, data_size_ * sizeof(int));
}

DataManager::~DataManager()
{
    if (data_)
    {
        free(data_);
        data_ = nullptr;
    }

    if (d_data_)
    {
        backend_->device_free(d_data_);
        d_data_ = nullptr;
    }
}

void DataManager::push_token(int token)
{
    if (current_index_ < data_size_)
    {
        data_[current_index_++] = token;
    }
    else
    {
        LOG_ERROR("DataManager buffer overflow. Cannot push more tokens.");
    }
}

void DataManager::save_data(const char *output_path)
{
    FILE *fp = fopen(output_path, "wb");

    if (!fp)
    {
        LOG_ERROR("Could not open file %s for writing.", output_path);
        return;
    }

    uint16_t *tmp_data = (uint16_t *)malloc(current_index_ * sizeof(uint16_t));

    if (!tmp_data)
    {
        LOG_ERROR("Could not allocate memory for saving data.");
        fclose(fp);
        return;
    }

    for (size_t i = 0; i < current_index_; ++i)
    {
        tmp_data[i] = (uint16_t)data_[i];
    }

    size_t count = fwrite(tmp_data, sizeof(uint16_t), current_index_, fp);

    fclose(fp);

    if (count != current_index_)
    {
        LOG_ERROR("Could not write entire tokens file %s.", output_path);
        return;
    }

    LOG_INFO("Saved tokens to %s successfully.", output_path);

    free(tmp_data);
}

const int *DataManager::device_data()
{
    backend_->device_memcpy_h2d(d_data_, data_, data_size_ * sizeof(int));

    return d_data_;
}

void DataManager::load_data(const char *data_path, int additional_tokens)
{
    FILE *fp = fopen(data_path, "rb");

    if (!fp)
    {
        LOG_ERROR("Could not open file %s for reading.", data_path);
        return;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    size_t count = size / sizeof(uint16_t);
    fseek(fp, 0, SEEK_SET);

    uint16_t *tmp_data = (uint16_t *)malloc(size);

    if (!tmp_data)
    {
        LOG_ERROR("Could not allocate memory for training data.");
        fclose(fp);
        return;
    }

    size_t read_count = fread(tmp_data, sizeof(uint16_t), count, fp);
    fclose(fp);

    if (read_count != count)
    {
        LOG_ERROR("Could not read all data from file %s.", data_path);
        free(tmp_data);
        return;
    }

    data_size_ = count + additional_tokens;
    current_index_ = count; // Start after the loaded data
    data_ = (int *)malloc(data_size_ * sizeof(int));

    if (!data_)
    {
        LOG_ERROR("Could not allocate memory for training data. (final)");
        free(tmp_data);
        return;
    }

    for (size_t i = 0; i < data_size_; ++i)
    {
        data_[i] = (int)tmp_data[i];
    }

    free(tmp_data);
    
    LOG_INFO("Loaded %zu tokens from %s", data_size_, data_path);
}