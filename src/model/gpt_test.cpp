#include "gpt.h"
#include "cpu_backend.h"
#include "cuda_backend.h"
#include "logger.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

static const char *TOKENS_FILE = "./data/gpt_dummy_tokens.bin";
static const char *WEIGHTS_FILE = "./data/gpt_dummy_weights.bin";
static const char *REF_GRADS_FILE = "./data/gpt_dummy_ref_gradients.bin";
static const float EPSILON = 1e-5f;

struct GradTensorInfo
{
    const char *name;
    size_t size;
};

static int compare_tensor(const char *name, const float *a, const float *b, size_t size, float eps)
{
    float max_diff = 0.0f;
    float max_a = 0.0f;
    size_t max_idx = 0;
    size_t num_bad = 0;

    for (size_t i = 0; i < size; ++i)
    {
        float diff = fabsf(a[i] - b[i]);

        if (diff > max_diff)
        {
            max_diff = diff;
            max_a = fabsf(a[i]);
            max_idx = i;
        }
        if (diff > eps || a[i] != a[i] || b[i] != b[i]) // check for NaN
        {
            ++num_bad;
        }
    }

    float rel_diff = max_a > 0.0f ? max_diff / max_a : 0.0f;

    if (num_bad == 0)
    {
        LOG_INFO("[ PASS ] %-14s size=%-9zu max_diff=%-14.3e rel_diff=%.3f%%", name, size, max_diff, rel_diff * 100.0f);
        return 1;
    }

    LOG_ERROR("[ FAIL ] %-14s size=%-9zu max_diff=%-14.3e rel_diff=%.3f%% at i=%zu (cpu=%.6f cuda=%.6f), %zu/%zu over eps", name, size, max_diff, rel_diff * 100.0f, max_idx, a[max_idx], b[max_idx], num_bad, size);
    return 0;
}

static int *upload_tokens(IGPTBackend *be, const int *host, int count)
{
    int *dev = NULL;
    be->device_malloc((void **)&dev, count * sizeof(int));
    be->device_memcpy_h2d(dev, host, count * sizeof(int));
    return dev;
}

// Fills the (name, size) table in the exact order gpt_weights packs its buffer.
static void build_grad_table(const gpt_config *c, GradTensorInfo *t)
{
    const size_t L = c->num_layers;
    const size_t S = c->max_seq_len;
    const size_t V = c->vocab_size;
    const size_t D = c->d_model;
    const size_t F = c->d_ffn;

    t[0].name = "wte";
    t[0].size = V * D;

    t[1].name = "wpe";
    t[1].size = S * D;

    t[2].name = "ln_1_w";
    t[2].size = L * D;

    t[3].name = "ln_1_b";
    t[3].size = L * D;

    t[4].name = "qkv_proj_w";
    t[4].size = L * D * 3 * D;

    t[5].name = "qkv_proj_b";
    t[5].size = L * 3 * D;

    t[6].name = "attn_proj_w";
    t[6].size = L * D * D;

    t[7].name = "attn_proj_b";
    t[7].size = L * D;

    t[8].name = "ln_2_w";
    t[8].size = L * D;

    t[9].name = "ln_2_b";
    t[9].size = L * D;

    t[10].name = "ffn_up_w";
    t[10].size = L * D * F;

    t[11].name = "ffn_up_b";
    t[11].size = L * F;

    t[12].name = "ffn_down_w";
    t[12].size = L * F * D;

    t[13].name = "ffn_down_b";
    t[13].size = L * D;

    t[14].name = "ln_f_w";
    t[14].size = D;

    t[15].name = "ln_f_b";
    t[15].size = D;
}

static int compare_models(GradTensorInfo *tensor_info, size_t count, const float *grads_1, const float *grads_2, const char *label_1, const char *label_2, float eps)
{
    LOG_INFO("-- weight gradient (%s vs %s) comparison (eps=%.1e) --", label_1, label_2, eps);

    int passed = 0;
    int failed = 0;
    size_t offset = 0;

    for (int i = 0; i < GPT_WEIGHTS_PARAMS_COUNT; ++i)
    {
        if (compare_tensor(tensor_info[i].name, grads_1 + offset, grads_2 + offset, tensor_info[i].size, eps))
        {
            ++passed;
        }
        else
        {
            ++failed;
        }

        offset += tensor_info[i].size;
    }

    LOG_INFO("--- %d passed, %d failed ---", passed, failed);

    return failed;
}

static float *run_and_download_grads(GPT *model, IGPTBackend *be, int *input_tokens, int *label_tokens)
{
    model->zero_grad(); // grads accumulate, must clear first
    model->forward(input_tokens);
    model->backward(input_tokens, label_tokens);
    float loss = model->loss(label_tokens); // compute loss for logging, but ignore the return value

    LOG_INFO("Loss: %.6f", loss);

    const gpt_weights *grads = model->grad_weights();
    size_t n = grads->buffer_count_;

    float *out = (float *)malloc(n * sizeof(float));
    be->device_memcpy_d2h(out, grads->buffer_, n * sizeof(float));

    return out;
}

int main()
{
    gpt_config config;
    config.max_seq_len = 64;
    config.vocab_size = 507;
    config.vocab_size_padded = 512;
    config.num_layers = 4;
    config.num_heads = 4;
    config.d_model = 32;
    config.d_ffn = 256;

    const int batch_size = 1;
    const int context = config.max_seq_len;

    LOG_INFO("GPT full forward/backward CPU-vs-CUDA test start");

    // Get tokens from file
    int *tokens = (int *)malloc((context + 1) * sizeof(int));

    {
        FILE *fp = fopen(TOKENS_FILE, "rb");

        if (fp)
        {
            fread(tokens, sizeof(int), context + 1, fp);
            fclose(fp);
            LOG_INFO("Read %d input tokens from %s.", context + 1, TOKENS_FILE);
        }
        else
        {
            LOG_ERROR("Could not open %s for reading tokens.", TOKENS_FILE);
        }
    }

    // Run CPU
    LOG_INFO("Running CPU model for reference gradients...");

    CPUBackend cpu_backend;
    GPT cpu_model(&cpu_backend, &config, true);
    cpu_model.load_checkpoint(WEIGHTS_FILE);
    cpu_model.set_size(batch_size, context);

    int *cpu_input = upload_tokens(&cpu_backend, tokens, context + 1);
    int *cpu_labels = cpu_input + 1; // labels are just the input shifted by 1
    float *cpu_grads = run_and_download_grads(&cpu_model, &cpu_backend, cpu_input, cpu_labels);

    cpu_backend.device_free(cpu_input);

    LOG_INFO("CPU weight gradients downloaded.");

    // RUN CUDA
    LOG_INFO("Running CUDA model for comparison...");

    CUDABackend cuda_backend;
    GPT cuda_model(&cuda_backend, &config, true);
    cuda_model.load_checkpoint(WEIGHTS_FILE);
    cuda_model.set_size(batch_size, context);

    int *cuda_input = upload_tokens(&cuda_backend, tokens, context + 1);
    int *cuda_labels = cuda_input + 1; // labels are just the input shifted by 1
    float *cuda_grads = run_and_download_grads(&cuda_model, &cuda_backend, cuda_input, cuda_labels);

    cuda_backend.device_free(cuda_input);

    // Get PyTorch gradients from file
    size_t pytorch_grads_size = cuda_model.grad_weights()->buffer_count_;
    float *pytorch_grads = (float *)malloc(pytorch_grads_size * sizeof(float));

    {
        FILE *fp = fopen(REF_GRADS_FILE, "rb");

        if (fp)
        {
            fread(pytorch_grads, sizeof(float), pytorch_grads_size, fp);
            fclose(fp);
            LOG_INFO("Read %d weights from %s.", (int)pytorch_grads_size, REF_GRADS_FILE);
        }
        else
        {
            LOG_ERROR("Could not open %s for reading weights.", REF_GRADS_FILE);
        }
    }

    // Run comparison
    LOG_INFO("CUDA weight gradients downloaded.");

    LOG_INFO("Comparing gradients...");

    GradTensorInfo table[GPT_WEIGHTS_PARAMS_COUNT];
    build_grad_table(&config, table); // Note: This assumes the the exact same order and no padding is used in the gpt_weights buffer.

    int failed = 0;
    failed += compare_models(table, GPT_WEIGHTS_PARAMS_COUNT, cpu_grads, cuda_grads, "CPU", "CUDA", EPSILON);
    failed += compare_models(table, GPT_WEIGHTS_PARAMS_COUNT, cuda_grads, pytorch_grads, "CUDA", "PyTorch", EPSILON);

    // Cleanup
    free(cpu_grads);
    free(cuda_grads);
    free(pytorch_grads);
    free(tokens);

    LOG_INFO("GPT full forward/backward CPU-vs-CUDA test finished.");

    return failed == 0 ? 0 : 1;
}