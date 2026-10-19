#include "gpt.h"
#include "cuda_backend.h"
#include "data_loader.h"
#include "lr_scheduling.h"
#include "logger.h"
#include "profile_range.h"
#include "timer.h"
#include <cstdlib>
#include <cstring>

int main(int argc, char *argv[])
{
    // Parse command-line arguments
    int epochs = -1;
    int max_steps = -1;
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
        else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc)
        {
            max_steps = atoi(argv[++i]);
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

    if (epochs <= 0 || max_steps == 0 || batch_size <= 0 || batch_accum_steps <= 0 || data_file == nullptr)
    {
        LOG_ERROR("Usage: %s --epochs <num_epochs> [--max-steps <optimizer_steps>] [--batch-size <batch_size>] [--batch-accum-steps <steps>] [--input <input_file>] [--output <output_file>] --data <data_file> [--seed <seed>]", argv[0]);
        return 1;
    }

    // Print configuration
    LOG_INFO("Training configuration: epochs=%d, max_steps=%d, batch_size=%d, batch_accum_steps=%d, input_file=%s, output_file=%s, data_file=%s, seed=%lu", epochs, max_steps, batch_size, batch_accum_steps, (input_file != nullptr) ? input_file : "None", (output_file != nullptr) ? output_file : "None", data_file, seed);

    if (output_file == nullptr)
    {
        LOG_WARN("No --output given, the trained weights will not be saved.");
    }

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
    const int tokens_per_step = batch_size * config.max_seq_len * batch_accum_steps;

    gpt.zero_grad();

    LOG_INFO("Starting training loop for %d epochs (%d optimizer steps)...", epochs, total_steps);

    Timer step_timer;

    // max_steps is -1 unless --max-steps is given, so global_steps never reaches it.
    for (int epoch = 0; epoch < epochs && global_steps != max_steps; epoch++)
    {
        loader.reset_epoch();

        while (global_steps != max_steps && loader.next_batch())
        {
            const int *input_tokens = loader.inputs();
            const int *label_tokens = loader.labels();

            {
                ProfileRange range(backend, "micro_step");

                gpt.forward(input_tokens);
                gpt.accumulate_loss(label_tokens);
                gpt.backward(input_tokens, label_tokens);
            }

            if (++micro % batch_accum_steps == 0)
            {
                ProfileRange range(backend, "update");

                float lr = get_lr_cosine_decay(global_steps++, 100, total_steps, 6e-4f, 6e-5f);

                gpt.unscale_and_clip_grads(1.0f, batch_accum_steps);
                gpt.optimizer_step(lr, 0.9f, 0.95f, 0.1f);
                gpt.zero_grad();

                // The step's only sync point: reading the loss waits for the GPU, so the clock is read after it.
                float loss = gpt.read_and_reset_loss();
                float step_ms = step_timer.elapsed_ms();
                step_timer.reset();

                LOG_INFO("Epoch %d/%d  |  Step %d/%d  |  Loss: %.6f  |  Step time: %.2f ms  |  %.0f tokens/s", epoch + 1, epochs, global_steps, total_steps, loss, step_ms, tokens_per_step * 1000.0f / step_ms);
            }
        }
    }

    // Finish training and save checkpoint
    LOG_INFO("Training completed.");

    if (output_file != nullptr && !gpt.save_checkpoint(output_file))
    {
        LOG_ERROR("Failed to save checkpoint.");
        return 1;
    }

    // Clean up
    delete backend;

    return 0;
}