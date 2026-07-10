#include "gpt.h"
#include "cuda_backend.h"
#include "data_manager.h"
#include "token_sampler.h"
#include "logger.h"
#include "benchmark.h"
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
    config.vocab_size_padded = 50304;
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
    TokenSampler sampler(config.vocab_size, temperature, top_k, top_p); // We use not padded vocab size for sampling, as the padded tokens are not valid tokens.

    if (!data_manager.load_data(input_file, max_tokens))
    {
        LOG_ERROR("Failed to load input data.");
        return 1;
    }

    float *new_token_logits_cpu = (float *)malloc(config.vocab_size_padded * sizeof(float));

    // Prepare stats
    float ttft_ms = 0.0f; // Time to first token
    float avg_forward_time_ms = 0.0f;
    float avg_sample_time_ms = 0.0f;

    // Generate tokens
    LOG_INFO("----------------------");

    for (int i = 0; i < max_tokens; ++i)
    {
        int current_size = data_manager.current_size();

        BENCHMARK_SCOPE_PRINT(Preparation, {
            gpt.set_size(1, current_size);
        });

        BENCHMARK_SCOPE_PRINT(ForwardPass, {
            // TODO: Add KV Cache (prefill + decode)
            gpt.forward(data_manager.device_data());
            backend->device_memcpy_d2h(new_token_logits_cpu, gpt.activations()->logits + (current_size - 1) * config.vocab_size_padded, config.vocab_size_padded * sizeof(float)); // Copy last token logits to host
        });

        BENCHMARK_SCOPE_PRINT(Sample, {
            data_manager.push_token(sampler.sample(new_token_logits_cpu));
        })

        if (i == 0)
        {
            ttft_ms = elapsed_ForwardPass + elapsed_Sample;
        }

        avg_forward_time_ms += elapsed_ForwardPass;
        avg_sample_time_ms += elapsed_Sample;

        LOG_INFO("----------------------");
    }

    // Print average times
    avg_forward_time_ms /= max_tokens;
    avg_sample_time_ms /= max_tokens;

    LOG_INFO("Time to first token (TTFT): %.2f ms", ttft_ms);
    LOG_INFO("Average forward pass time: %.2f ms (%.2f tokens/second)", avg_forward_time_ms, 1000.0f / avg_forward_time_ms);
    LOG_INFO("Average sampling time: %.2f ms (%.2f tokens/second)", avg_sample_time_ms, 1000.0f / avg_sample_time_ms);

    // Save generated tokens to output file
    if (!data_manager.save_data(output_file))
    {
        LOG_ERROR("Failed to save generated tokens.");
        return 1;
    }

    // Clean up
    delete backend;
    free(new_token_logits_cpu);

    return 0;
}