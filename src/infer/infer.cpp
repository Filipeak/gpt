#include "gpt.h"
#include "cuda_backend.h"
#include "data_manager.h"
#include "token_sampler.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char *argv[])
{
    // Parse command-line arguments
    char *model_file = nullptr;
    char *input_file = nullptr;
    char *output_file = nullptr;
    unsigned long seed = 42;
    int max_tokens = 128;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
        {
            model_file = argv[++i];
        }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
        {
            input_file = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            output_file = argv[++i];
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed = strtoul(argv[++i], nullptr, 10);
        }
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
        {
            max_tokens = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
        {
            temperature = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
        {
            top_k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
        {
            top_p = atof(argv[++i]);
        }
    }

    if (model_file == nullptr || input_file == nullptr || output_file == nullptr)
    {
        printf("Usage: %s --model <model_file> --input <input_file> --output <output_file> [--seed <seed>] [--max-tokens <max_tokens>] [--temperature <temperature>] [--top-k <top_k>] [--top-p <top_p>]\n", argv[0]);
        return 1;
    }

    // Print configuration
    printf("Inference configuration:\n");
    printf("- model_file=%s\n", model_file);
    printf("- input_file=%s\n", input_file);
    printf("- output_file=%s\n", output_file);
    printf("- seed=%lu\n", seed);
    printf("- max_tokens=%d\n", max_tokens);
    printf("- temperature=%.2f\n", temperature);
    printf("- top_k=%d\n", top_k);
    printf("- top_p=%.2f\n", top_p);

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
    GPT gpt(backend, &config, false);

    // Load checkpoint
    gpt.load_checkpoint(model_file);

    // Prepare data
    DataManager data_manager(backend, input_file, max_tokens);
    float *new_token_logits = (float *)malloc(config.vocab_size * sizeof(float));

    // Measure time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Generate tokens
    for (int i = 0; i < max_tokens; ++i)
    {
        int current_size = data_manager.current_size();

        gpt.set_size(1, current_size);
        gpt.forward(data_manager.device_data());

        const float *logits = gpt.activations()->logits;
        backend->device_memcpy_d2h(new_token_logits, logits + (current_size - 1) * config.vocab_size, config.vocab_size * sizeof(float)); // Copy last token logits to host

        int next_token = sample_token(new_token_logits, config.vocab_size, temperature, top_k, top_p);

        printf("Generated token %d/%d: %d\n", i + 1, max_tokens, next_token);

        data_manager.push_token(next_token);
    }

    // Measure time
    auto end_time = std::chrono::high_resolution_clock::now();
    printf("Generated %d tokens in %.2f seconds. (%.2f tokens/second)\n", max_tokens, std::chrono::duration<double>(end_time - start_time).count(), (double)max_tokens / std::chrono::duration<double>(end_time - start_time).count());

    // Save generated tokens to output file
    data_manager.save_data(output_file);

    // Clean up
    delete backend;
    free(new_token_logits);

    return 0;
}