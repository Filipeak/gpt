#include "gpt.h"
#include "cuda_backend.h"
#include "data_loader.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

float get_lr(int step, int warmup, int total, float max_lr, float min_lr)
{
    if (step < warmup) // linear warmup
    {
        return max_lr * (step + 1) / warmup;
    }

    if (step >= total)
    {
        return min_lr;
    }

    float prog = (float)(step - warmup) / (total - warmup); // 0..1

    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + cosf(3.14159265f * prog));
}

int main(int argc, char *argv[])
{
    // Parse command-line arguments
    int epochs = -1;
    int batch_size = 2;
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

    if (epochs <= 0 || output_file == nullptr || data_file == nullptr)
    {
        printf("Usage: %s --epochs <num_epochs> --batch-size <batch_size> [--input <input_file>] --output <output_file> [--data <data_file>] [--seed <seed>]\n", argv[0]);
        return 1;
    }

    // Print configuration
    printf("Training configuration:\n");
    printf("- Epochs: %d\n", epochs);
    printf("- Batch size: %d\n", batch_size);
    printf("- Input checkpoint: %s\n", (input_file != nullptr) ? input_file : "None");
    printf("- Output checkpoint: %s\n", output_file);
    printf("- Data file: %s\n", data_file);
    printf("- Seed: %lu\n", seed);

    // Prepare config
    gpt_config config;
    config.max_seq_len = 256;
    config.vocab_size = 50257;
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
        gpt.load_checkpoint(input_file);
    }
    else
    {
        gpt.init(0.0f, 0.02f);
    }

    // Prepare data
    DataLoader loader(backend, data_file, batch_size, config.max_seq_len);
    gpt.set_size(batch_size, config.max_seq_len);

    // Print memory info
    float free_mem_mb, total_mem_mb;
    backend->device_get_memory_info(&free_mem_mb, &total_mem_mb);
    printf("Memory Info: Free: %.2f MB, Total: %.2f MB\n", free_mem_mb, total_mem_mb);

    // Training loop
    const int accum_steps = 2;
    int micro = 0;
    int global_steps = 0;
    int total_steps = epochs * loader.total_batches() / accum_steps;

    gpt.zero_grad();

    printf("Starting training loop for %d epochs...\n", epochs);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        loader.reset_epoch();

        while (loader.next_batch())
        {
            const int *input_tokens = loader.inputs();
            const int *label_tokens = loader.labels();

            gpt.forward(input_tokens);
            float loss = gpt.loss(label_tokens);
            gpt.backward(input_tokens, label_tokens);

            if (++micro % accum_steps == 0)
            {
                float lr = get_lr(global_steps++, 100, total_steps, 1.2e-3f, 1.2e-4f);

                gpt.clip_grad_norm(1.0f);
                gpt.optimizer_step(lr, 0.9f, 0.999f, 0.1f);
                gpt.zero_grad();
                
                printf("Epoch %d/%d  |  Batch %d/%d  |  Loss: %.6f\n", epoch + 1, epochs, loader.current_batch(), loader.total_batches(), loss);
            }
        }
    }

    // Finish training and save checkpoint
    printf("Training completed.\n");

    gpt.save_checkpoint(output_file);

    // Clean up
    delete backend;

    return 0;
}