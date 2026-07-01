#include "gpt.h"
#include "cuda_backend.h"
#include "data_manager.h"
#include "token_sampler.h"
#include "logger.h"
#include <chrono>
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

    if (model_file == nullptr || input_file == nullptr || output_file == nullptr || max_tokens <= 0 || temperature <= 0.0f || top_k < 0 || top_p < 0.0f || top_p > 1.0f)
    {
        LOG_ERROR("Usage: %s --model <model_file> --input <input_file> --output <output_file> [--seed <seed>] [--max-tokens <max_tokens>] [--temperature <temperature>] [--top-k <top_k>] [--top-p <top_p>]\n", argv[0]);
        return 1;
    }

    // Print configuration
    LOG_INFO("Inference configuration: model_file=%s, input_file=%s, output_file=%s, seed=%lu, max_tokens=%d, temperature=%.2f, top_k=%d, top_p=%.2f", model_file, input_file, output_file, seed, max_tokens, temperature, top_k, top_p);

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
    if (!gpt.load_checkpoint(model_file))
    {
        LOG_ERROR("Failed to load checkpoint.");
        return 1;
    }

    // Prepare data
    DataManager data_manager(backend);

    if (!data_manager.load_data(input_file, max_tokens))
    {
        LOG_ERROR("Failed to load input data.");
        return 1;
    }

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

        LOG_DEBUG("Generated token %d/%d: %d", i + 1, max_tokens, next_token);

        data_manager.push_token(next_token);
    }

    // Measure time
    auto end_time = std::chrono::high_resolution_clock::now();
    LOG_INFO("Generated %d tokens in %.2f seconds. (%.2f tokens/second)", max_tokens, std::chrono::duration<double>(end_time - start_time).count(), (double)max_tokens / std::chrono::duration<double>(end_time - start_time).count());

    // Save generated tokens to output file
    if (!data_manager.save_data(output_file))
    {
        LOG_ERROR("Failed to save generated tokens.");
        return 1;
    }

    // Clean up
    delete backend;
    free(new_token_logits);

    return 0;
}