#include "gpt.h"
#include "cuda_backend.h"
#include "data_loader.h"
#include "lr_scheduling.h"
#include "logger.h"
#include "benchmark.h"
#include <cstdlib>
#include <cstring>

int main(int argc, char *argv[])
{
    // Parse command-line arguments
    int epochs = -1;
    int batch_size = 2;
    int batch_accum_steps = 2;
    char *input_file = nullptr;
    char *output_file = nullptr;
    char *data_file = nullptr;
    unsigned long seed = 42;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
        {
            epochs = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
        {
            batch_size = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--batch-accum-steps") == 0 && i + 1 < argc)
        {
            batch_accum_steps = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
        {
            input_file = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            output_file = argv[++i];
        }
        else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
        {
            data_file = argv[++i];
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed = strtoul(argv[++i], nullptr, 10);
        }
    }

    if (epochs <= 0 || batch_size <= 0 || batch_accum_steps <= 0 || output_file == nullptr || data_file == nullptr)
    {
        LOG_ERROR("Usage: %s --epochs <num_epochs> [--batch-size <batch_size>] [--batch-accum-steps <steps>] [--input <input_file>] --output <output_file> [--data <data_file>] [--seed <seed>]", argv[0]);
        return 1;
    }

    // Print configuration
    LOG_INFO("Training configuration: epochs=%d, batch_size=%d, batch_accum_steps=%d, input_file=%s, output_file=%s, data_file=%s, seed=%lu", epochs, batch_size, batch_accum_steps, (input_file != nullptr) ? input_file : "None", output_file, data_file, seed);

    // Prepare config
    gpt_config config;
    config.max_seq_len = 256;
    config.vocab_size = 50257;
    config.vocab_size_padded = 50304;
    config.num_layers = 12;
    config.num_heads = 12;
    config.d_model = 768;
    config.d_ffn = 3072;

    // Initialize backend and GPT model
    srand(seed);

    IGPTBackend *backend = new CUDABackend(seed);
    GPT gpt(backend, &config, true);

    // Load checkpoint if provided, otherwise initialize weights
    if (input_file != nullptr)
    {
        if (!gpt.load_checkpoint(input_file))
        {
            return 1;
        }
    }
    else
    {
        gpt.init(0.0f, 0.02f);
    }

    // Prepare data
    DataLoader loader(backend, batch_size, config.max_seq_len);
    if (!loader.load_data(data_file))
    {
        LOG_ERROR("Failed to load data from %s.", data_file);
        return 1;
    }

    gpt.set_size(batch_size, config.max_seq_len);

    // Print memory info
    float free_mem_mb, total_mem_mb;
    backend->device_get_memory_info(&free_mem_mb, &total_mem_mb);
    LOG_INFO("Memory Info: Free: %.2f MB, Total: %.2f MB", free_mem_mb, total_mem_mb);

    // Training loop
    int micro = 0;
    int global_steps = 0;
    int total_steps = epochs * loader.total_batches() / batch_accum_steps;

    float running_training_step_time_ms = 0.0f;
    float running_loss = 0.0f;
    int running_count = 0;

    gpt.zero_grad();

    LOG_INFO("Starting training loop for %d epochs...", epochs);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        loader.reset_epoch();

        while (loader.next_batch())
        {
            const int *input_tokens = loader.inputs();
            const int *label_tokens = loader.labels();

            BENCHMARK_SCOPE(TrainingStep, {
                gpt.forward(input_tokens);
                gpt.backward(input_tokens, label_tokens);
                float loss = gpt.loss(label_tokens);
                running_loss += loss;
            });

            running_training_step_time_ms += elapsed_TrainingStep;
            running_count++;

            if (++micro % batch_accum_steps == 0)
            {
                float lr = get_lr_cosine_decay(global_steps++, 100, total_steps, 6e-4f, 6e-5f);

                gpt.unscale_and_clip_grads(1.0f, batch_accum_steps);
                gpt.optimizer_step(lr, 0.9f, 0.95f, 0.1f);
                gpt.zero_grad();

                LOG_INFO("Epoch %d/%d  |  Batch %d/%d  |  Avg Loss: %.6f  |  Avg Time: %.2f ms", epoch + 1, epochs, loader.current_batch(), loader.total_batches(), running_loss / running_count, running_training_step_time_ms / running_count);

                running_training_step_time_ms = 0.0f;
                running_loss = 0.0f;
                running_count = 0;
            }
        }
    }

    // Finish training and save checkpoint
    LOG_INFO("Training completed.");

    if (!gpt.save_checkpoint(output_file))
    {
        LOG_ERROR("Failed to save checkpoint.");
        return 1;
    }

    // Clean up
    delete backend;

    return 0;
}