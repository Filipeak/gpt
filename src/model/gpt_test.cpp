#include "gpt.h"
#include "cpu_backend.h"
#include "cuda_backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

static const char *CHECKPOINT_FILE = "./data/gpt_dummy.bin";
static const float EPSILON = 1e-8f;

struct GradTensorInfo
{
    const char *name;
    size_t size;
};

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");

    if (f)
    {
        fclose(f);
        return 1;
    }

    return 0;
}

static int compare_tensor(const char *name, const float *a, const float *b, size_t size, float eps)
{
    float max_diff = 0.0f;
    size_t max_idx = 0;
    size_t num_bad = 0;

    for (size_t i = 0; i < size; ++i)
    {
        float diff = fabsf(a[i] - b[i]);

        if (diff > max_diff)
        {
            max_diff = diff;
            max_idx = i;
        }
        if (diff > eps)
        {
            ++num_bad;
        }
    }

    if (num_bad == 0)
    {
        printf("[ PASS ] %-14s size=%-9zu max_diff=%.3e\n", name, size, max_diff);
        return 1;
    }

    printf("[ FAIL ] %-14s size=%-9zu max_diff=%.3e at i=%zu (cpu=%.6f cuda=%.6f), %zu/%zu over eps\n", name, size, max_diff, max_idx, a[max_idx], b[max_idx], num_bad, size);
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

static float *run_and_download_grads(GPT *model, IGPTBackend *be, int *input_tokens, int *label_tokens, size_t *count_out)
{
    model->zero_grad(); // grads accumulate, must clear first
    model->forward(input_tokens);
    model->backward(input_tokens, label_tokens);

    const gpt_weights *grads = model->grad_weights();
    size_t n = grads->buffer_count_;

    float *out = (float *)malloc(n * sizeof(float));
    be->device_memcpy_d2h(out, grads->buffer_, n * sizeof(float));

    *count_out = n;
    return out;
}

int main()
{
    gpt_config config;
    config.max_seq_len = 16;
    config.vocab_size = 32;
    config.num_layers = 2;
    config.num_heads = 2;
    config.d_model = 8;
    config.d_ffn = 32;

    const int batch_size = 1;
    const int context = config.max_seq_len;

    printf("> GPT full forward/backward CPU-vs-CUDA test start\n");

    // Dummy tokens: random list of size context + 1.
    // input = tokens[0:context], labels = tokens[1:context+1].
    srand(1234);

    int *tokens = (int *)malloc((context + 1) * sizeof(int));

    for (int i = 0; i < context + 1; ++i)
    {
        tokens[i] = rand() % config.vocab_size;
    }

    CPUBackend cpu_backend;
    GPT cpu_model(&cpu_backend, &config, true);

    // Create the checkpoint if it does not exist yet.
    if (!file_exists(CHECKPOINT_FILE))
    {
        printf("Checkpoint %s not found, creating dummy model.\n", CHECKPOINT_FILE);
        cpu_model.init(0.0f, 0.02f);
        cpu_model.save_checkpoint(CHECKPOINT_FILE);
    }

    cpu_model.load_checkpoint(CHECKPOINT_FILE);
    cpu_model.set_size(batch_size, context);

    int *cpu_input = upload_tokens(&cpu_backend, tokens, context + 1);
    int *cpu_labels = cpu_input + 1; // labels are just the input shifted by 1

    size_t cpu_count = 0;
    float *cpu_grads = run_and_download_grads(&cpu_model, &cpu_backend, cpu_input, cpu_labels, &cpu_count);

    cpu_backend.device_free(cpu_input);

    printf("> CPU weight gradients downloaded.\n"
           "> Now running CUDA model for comparison...\n");

    // Now run the same model on CUDA and compare the gradients.
    CUDABackend cuda_backend;
    GPT cuda_model(&cuda_backend, &config, true);
    cuda_model.load_checkpoint(CHECKPOINT_FILE);
    cuda_model.set_size(batch_size, context);

    int *cuda_input = upload_tokens(&cuda_backend, tokens, context + 1);
    int *cuda_labels = cuda_input + 1; // labels are just the input shifted by 1

    size_t cuda_count = 0;
    float *cuda_grads = run_and_download_grads(&cuda_model, &cuda_backend, cuda_input, cuda_labels, &cuda_count);

    cuda_backend.device_free(cuda_input);

    printf("> CUDA weight gradients downloaded.\n"
           "> Now comparing gradients...\n");

    printf("-- weight gradient comparison (eps=%.1e) --\n", EPSILON);

    int failed = 0;

    if (cpu_count != cuda_count)
    {
        printf("Error: gradient buffer size mismatch (cpu=%zu, cuda=%zu)\n", cpu_count, cuda_count);
        failed = 1;
    }
    else
    {
        GradTensorInfo table[GPT_WEIGHTS_PARAMS_COUNT];
        build_grad_table(&config, table);

        int passed = 0;
        size_t offset = 0;

        for (int i = 0; i < GPT_WEIGHTS_PARAMS_COUNT; ++i)
        {
            if (compare_tensor(table[i].name, cpu_grads + offset, cuda_grads + offset, table[i].size, EPSILON))
            {
                ++passed;
            }
            else
            {
                ++failed;
            }

            offset += table[i].size;
        }

        printf("--- %d passed, %d failed ---\n", passed, failed);
    }

    free(cpu_grads);
    free(cuda_grads);
    free(tokens);

    printf("> GPT full forward/backward CPU-vs-CUDA test finished.\n");

    return failed == 0 ? 0 : 1;
}