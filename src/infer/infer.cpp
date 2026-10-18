#include "gpt.h"
#include "cuda_backend.h"
#include "data_manager.h"
#include "token_sampler.h"
#include "tokenizer.h"
#include "unicode.h"
#include "logger.h"
#include "benchmark.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

// Windows passes argv in the ANSI code page, which mangles non-ASCII prompts,
// so the argument is read again from the UTF-16 command line and converted to UTF-8.
static std::string utf8_arg(int argc, char *argv[], int index)
{
    std::string arg = argv[index];
    int wargc = 0;
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);

    if (wargv && wargc == argc)
    {
        int wlen = (int)wcslen(wargv[index]);
        int len = WideCharToMultiByte(CP_UTF8, 0, wargv[index], wlen, nullptr, 0, nullptr, nullptr);
        arg.assign(len, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wargv[index], wlen, arg.data(), len, nullptr, nullptr);
    }

    LocalFree(wargv);
    return arg;
}

// Switches the console to UTF-8 so the generated text shows correctly, and restores it on exit.
struct Utf8Console
{
    UINT previous_cp = GetConsoleOutputCP();
    Utf8Console() { SetConsoleOutputCP(CP_UTF8); }
    ~Utf8Console() { SetConsoleOutputCP(previous_cp); }
};
#else
static std::string utf8_arg(int argc, char *argv[], int index) { return argv[index]; }
struct Utf8Console
{
};
#endif

int main(int argc, char *argv[])
{
    Utf8Console console;

    // Parse command-line arguments
    char *model_file = nullptr;
    char *input_file = nullptr;
    char *output_file = nullptr;
    const char *vocab_file = "./data/gpt2_vocab.bin";
    std::string prompt;
    bool has_prompt = false;
    bool verbose = false;
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
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
        {
            prompt = utf8_arg(argc, argv, ++i);
            has_prompt = true;
        }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
        {
            input_file = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            output_file = argv[++i];
        }
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
        {
            vocab_file = argv[++i];
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
        else if (strcmp(argv[i], "--verbose") == 0)
        {
            verbose = true;
        }
    }

    bool one_prompt_source = has_prompt != (input_file != nullptr);

    if (model_file == nullptr || !one_prompt_source || max_tokens <= 0 || temperature <= 0.0f || top_k < 0 || top_p < 0.0f || top_p > 1.0f)
    {
        LOG_ERROR("Usage: %s --model <model_file> (--prompt <text> | --input <input_file>) [--output <output_file>] [--vocab <vocab_file>] [--seed <seed>] [--max-tokens <max_tokens>] [--temperature <temperature>] [--top-k <top_k>] [--top-p <top_p>] [--verbose]\n", argv[0]);
        return 1;
    }

    // Print configuration
    LOG_INFO("Inference configuration: model_file=%s, input=%s, output_file=%s, vocab_file=%s, seed=%lu, max_tokens=%d, temperature=%.2f, top_k=%d, top_p=%.2f", model_file, has_prompt ? "--prompt" : input_file, output_file ? output_file : "(none)", vocab_file, seed, max_tokens, temperature, top_k, top_p);

    // Load tokenizer
    Tokenizer tokenizer;

    if (!tokenizer.load(vocab_file))
    {
        LOG_ERROR("Failed to load vocab. Create it with: python data/tokenizer.py export-vocab %s", vocab_file);
        return 1;
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

    bool data_loaded = has_prompt ? data_manager.load_tokens(tokenizer.encode(prompt), max_tokens)
                                  : data_manager.load_data(input_file, max_tokens);

    if (!data_loaded)
    {
        LOG_ERROR("Failed to load input data.");
        return 1;
    }

    int prompt_size = (int)data_manager.current_size();

    if (prompt_size == 0)
    {
        LOG_ERROR("The prompt is empty.");
        return 1;
    }

    if (prompt_size + max_tokens > config.max_seq_len)
    {
        LOG_ERROR("Prompt (%d tokens) + max tokens (%d) does not fit in the model's context of %d tokens.", prompt_size, max_tokens, config.max_seq_len);
        return 1;
    }

    LOG_INFO("Prompt: %d tokens.", prompt_size);

    // Prints tokens as text as soon as they are generated. Bytes of a UTF-8 character
    // split across tokens are held back until the character is complete. With --verbose
    // the per-step logs would break up the text, so it is all printed at the end instead.
    std::string pending_text;

    auto print_token = [&](int token)
    {
        pending_text += tokenizer.decode(token);

        if (verbose)
        {
            return;
        }

        size_t printable = utf8_complete_length(pending_text);
        fwrite(pending_text.data(), 1, printable, stdout);
        fflush(stdout);
        pending_text.erase(0, printable);
    };

    float *new_token_logits_cpu = (float *)malloc(config.vocab_size_padded * sizeof(float));
    const int max_warmup_steps = 5;
    int remaining_warmup_steps = max_warmup_steps;

    // Prepare stats
    float ttft_ms = 0.0f; // Time to first token
    float avg_forward_time_ms = 0.0f;
    float avg_sample_time_ms = 0.0f;

    // Generate tokens, echoing the prompt first
    LOG_INFO("----------------------");

    for (int i = 0; i < prompt_size; ++i)
    {
        print_token(data_manager.host_data()[i]);
    }

    for (int i = 0; i < max_tokens; ++i)
    {
        int current_size = data_manager.current_size();

        BENCHMARK_SCOPE(Preparation, {
            gpt.set_size(1, current_size);
        });

        BENCHMARK_SCOPE(ForwardPass, {
            // TODO: Add KV Cache (prefill + decode)
            gpt.forward(data_manager.device_data());
            backend->device_memcpy_d2h(new_token_logits_cpu, gpt.activations()->logits + (current_size - 1) * config.vocab_size_padded, config.vocab_size_padded * sizeof(float)); // Copy last token logits to host
        });

        if (verbose)
        {
            LOG_INFO("Preparation took %.2f ms", elapsed_Preparation);
            LOG_INFO("ForwardPass took %.2f ms", elapsed_ForwardPass);
        }

        if (remaining_warmup_steps > 0)
        {
            --remaining_warmup_steps;
            --i;

            if (verbose)
            {
                LOG_INFO("Warmup step %d/%d completed.", max_warmup_steps - remaining_warmup_steps, max_warmup_steps);
            }
        }
        else
        {
            int token = 0;

            BENCHMARK_SCOPE(Sample, {
                token = sampler.sample(new_token_logits_cpu);
            });

            data_manager.push_token(token);
            print_token(token);

            if (verbose)
            {
                LOG_INFO("Sample took %.2f ms", elapsed_Sample);
            }

            if (i == 0)
            {
                ttft_ms = elapsed_ForwardPass + elapsed_Sample;
            }

            avg_forward_time_ms += elapsed_ForwardPass;
            avg_sample_time_ms += elapsed_Sample;
        }
    }

    fwrite(pending_text.data(), 1, pending_text.size(), stdout);
    printf("\n");
    LOG_INFO("----------------------");

    // Print average times
    avg_forward_time_ms /= max_tokens;
    avg_sample_time_ms /= max_tokens;

    LOG_INFO("Time to first token (TTFT): %.2f ms", ttft_ms);
    LOG_INFO("Average forward pass time: %.2f ms (%.2f tokens/second)", avg_forward_time_ms, 1000.0f / avg_forward_time_ms);
    LOG_INFO("Average sampling time: %.2f ms (%.2f tokens/second)", avg_sample_time_ms, 1000.0f / avg_sample_time_ms);

    // Save generated tokens to output file
    if (output_file != nullptr && !data_manager.save_data(output_file))
    {
        LOG_ERROR("Failed to save generated tokens.");
        return 1;
    }

    // Clean up
    delete backend;
    free(new_token_logits_cpu);

    return 0;
}