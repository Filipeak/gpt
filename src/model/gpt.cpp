#include "gpt.h"
#include "logger.h"
#include <cstdlib>
#include <cmath>

#define PAD_SIZE_4(size) (((size) + 3) & ~3) // Round up to nearest multiple of 4

void gpt_config::print() const
{
    LOG_INFO("GPT Configuration: max_seq_len=%d, vocab_size=%d (padded: %d), num_layers=%d, num_heads=%d, d_model=%d, d_ffn=%d", max_seq_len, vocab_size, vocab_size_padded, num_layers, num_heads, d_model, d_ffn);
}

static float *allocate_tensor_struct(IGPTBackend *backend, float **tensor_ptr[], size_t *sizes, size_t num_tensors, bool pad_4, size_t *total_size_out)
{
    size_t total_size = 0;

    for (size_t i = 0; i < num_tensors; ++i)
    {
        total_size += pad_4 ? PAD_SIZE_4(sizes[i]) : sizes[i];
    }

    float *ptr;
    backend->device_malloc((void **)&ptr, total_size * sizeof(float));

    float *ptr_iter = ptr;

    for (size_t i = 0; i < num_tensors; ++i)
    {
        *(tensor_ptr[i]) = ptr_iter;
        ptr_iter += pad_4 ? PAD_SIZE_4(sizes[i]) : sizes[i];
    }

    *total_size_out = total_size;

    return ptr;
}

gpt_weights::gpt_weights(IGPTBackend *backend, const gpt_config *config) : backend_(backend), buffer_(nullptr), buffer_count_(0)
{
    const int L = config->num_layers;
    const int S = config->max_seq_len;
    const int Vp = config->vocab_size_padded;
    const int D = config->d_model;
    const int F = config->d_ffn;

    size_t sizes[GPT_WEIGHTS_PARAMS_COUNT];
    sizes[0] = Vp * D;        // wte
    sizes[1] = S * D;         // wpe
    sizes[2] = L * D;         // ln_1_w
    sizes[3] = L * D;         // ln_1_b
    sizes[4] = L * D * 3 * D; // qkv_proj_w
    sizes[5] = L * 3 * D;     // qkv_proj_b
    sizes[6] = L * D * D;     // attn_proj_w
    sizes[7] = L * D;         // attn_proj_b
    sizes[8] = L * D;         // ln_2_w
    sizes[9] = L * D;         // ln_2_b
    sizes[10] = L * D * F;    // ffn_up_w
    sizes[11] = L * F;        // ffn_up_b
    sizes[12] = L * F * D;    // ffn_down_w
    sizes[13] = L * D;        // ffn_down_b
    sizes[14] = D;            // ln_f_w
    sizes[15] = D;            // ln_f_b

    float **ptrs[] = {
        &this->wte,
        &this->wpe,
        &this->ln_1_w,
        &this->ln_1_b,
        &this->qkv_proj_w,
        &this->qkv_proj_b,
        &this->attn_proj_w,
        &this->attn_proj_b,
        &this->ln_2_w,
        &this->ln_2_b,
        &this->ffn_up_w,
        &this->ffn_up_b,
        &this->ffn_down_w,
        &this->ffn_down_b,
        &this->ln_f_w,
        &this->ln_f_b,
    };

    buffer_ = allocate_tensor_struct(backend_, ptrs, sizes, GPT_WEIGHTS_PARAMS_COUNT, false, &buffer_count_); // Weights do not need padding, because they do not depend on batch size or sequence length, and are already aligned by design.

    LOG_DEBUG("Allocated GPT weights buffer of size %zu floats (%.2f MB).", buffer_count_, buffer_count_ * sizeof(float) / (1024.0 * 1024.0));
}

gpt_weights::~gpt_weights()
{
    if (buffer_)
    {
        backend_->device_free(buffer_);
        buffer_ = nullptr;

        LOG_DEBUG("GPT weights buffer freed. (count: %zu)", buffer_count_);

        buffer_count_ = 0;
    }
}

gpt_activations::gpt_activations(IGPTBackend *backend, const gpt_config *config, int batch_size, int seq_len) : backend_(backend), buffer_(nullptr), buffer_count_(0)
{
    const int L = config->num_layers;
    const int B = batch_size;
    const int T = seq_len;
    const int Vp = config->vocab_size_padded;
    const int H = config->num_heads;
    const int D = config->d_model;
    const int F = config->d_ffn;

    size_t sizes[GPT_ACTIVATIONS_COUNT];
    sizes[0] = B * T * D;         // x_emb
    sizes[1] = L * B * T * D;     // ln_1_out
    sizes[2] = L * B * T;         // ln_1_means
    sizes[3] = L * B * T;         // ln_1_vars
    sizes[4] = L * B * T * 3 * D; // qkv_out
    sizes[5] = L * B * T * H;     // attn_logsumexp
    sizes[6] = L * B * T * D;     // attn_out
    sizes[7] = L * B * T * D;     // attn_proj
    sizes[8] = L * B * T * D;     // ln_2_out
    sizes[9] = L * B * T;         // ln_2_means
    sizes[10] = L * B * T;        // ln_2_vars
    sizes[11] = L * B * T * F;    // ffn_up
    sizes[12] = L * B * T * F;    // ffn_act
    sizes[13] = L * B * T * D;    // ffn_down
    sizes[14] = B * T * D;        // ln_f_out
    sizes[15] = B * T;            // ln_f_means
    sizes[16] = B * T;            // ln_f_vars
    sizes[17] = B * T * Vp;       // logits
    sizes[18] = B * T * Vp;       // probs

    float **ptrs[] = {
        &this->x_emb,
        &this->ln_1_out,
        &this->ln_1_means,
        &this->ln_1_vars,
        &this->qkv_out,
        &this->attn_logsumexp,
        &this->attn_out,
        &this->attn_proj,
        &this->ln_2_out,
        &this->ln_2_means,
        &this->ln_2_vars,
        &this->ffn_up,
        &this->ffn_act,
        &this->ffn_down,
        &this->ln_f_out,
        &this->ln_f_means,
        &this->ln_f_vars,
        &this->logits,
        &this->probs,
    };

    buffer_ = allocate_tensor_struct(backend_, ptrs, sizes, GPT_ACTIVATIONS_COUNT, true, &buffer_count_);

    LOG_DEBUG("Allocated GPT activations buffer of size %zu floats (%.2f MB) for batch size %d and sequence length %d.", buffer_count_, buffer_count_ * sizeof(float) / (1024.0 * 1024.0), batch_size, seq_len);
}

gpt_activations::~gpt_activations()
{
    if (buffer_)
    {
        backend_->device_free(buffer_);
        buffer_ = nullptr;

        LOG_DEBUG("GPT activations buffer freed. (count: %zu)", buffer_count_);

        buffer_count_ = 0;
    }
}

gpt_cache_x_grads::gpt_cache_x_grads(IGPTBackend *backend, const gpt_config *config, int batch_size, int seq_len) : backend_(backend), buffer_(nullptr), buffer_count_(0)
{
    const int L = config->num_layers;
    const int B = batch_size;
    const int T = seq_len;
    const int Vp = config->vocab_size_padded;
    const int H = config->num_heads;
    const int D = config->d_model;
    const int F = config->d_ffn;

    size_t sizes[GPT_CACHE_X_GRADS_COUNT];
    sizes[0] = L * B * T * D;     // ln_1
    sizes[1] = L * B * T * D;     // qkv_proj
    sizes[2] = L * B * T * 3 * D; // attn
    sizes[3] = L * B * T * H;     // attn_d_helper
    sizes[4] = L * B * T * D;     // attn_proj
    sizes[5] = L * B * T * D;     // ln_2
    sizes[6] = L * B * T * D;     // ffn_up
    sizes[7] = L * B * T * F;     // ffn_act
    sizes[8] = L * B * T * F;     // ffn_down
    sizes[9] = B * T * D;         // ln_f
    sizes[10] = B * T * D;        // unembedding
    sizes[11] = B * T * Vp;       // softmax_final

    float **ptrs[] = {
        &this->ln_1,
        &this->qkv_proj,
        &this->attn,
        &this->attn_d_helper,
        &this->attn_proj,
        &this->ln_2,
        &this->ffn_up,
        &this->ffn_act,
        &this->ffn_down,
        &this->ln_f,
        &this->unembedding,
        &this->softmax_final,
    };

    buffer_ = allocate_tensor_struct(backend_, ptrs, sizes, GPT_CACHE_X_GRADS_COUNT, true, &buffer_count_);

    LOG_DEBUG("Allocated GPT cache_x_grads buffer of size %zu floats (%.2f MB) for batch size %d and sequence length %d.", buffer_count_, buffer_count_ * sizeof(float) / (1024.0 * 1024.0), batch_size, seq_len);
}

gpt_cache_x_grads::~gpt_cache_x_grads()
{
    if (buffer_)
    {
        backend_->device_free(buffer_);
        buffer_ = nullptr;

        LOG_DEBUG("GPT cache_x_grads buffer freed. (count: %zu)", buffer_count_);

        buffer_count_ = 0;
    }
}

GPT::GPT(IGPTBackend *backend, const gpt_config *config, bool use_grad) : backend_(backend), config_(config), use_grad_(use_grad)
{
    config_->print();

    weights_ = new gpt_weights(backend_, config_);

    if (use_grad_)
    {
        weights_grads_ = new gpt_weights(backend_, config_);
        weights_grads_momentum_ = new gpt_weights(backend_, config_);
        weights_grads_velocity_ = new gpt_weights(backend_, config_);

        backend_->device_memset(weights_grads_momentum_->buffer_, 0, weights_grads_momentum_->buffer_count_ * sizeof(float));
        backend_->device_memset(weights_grads_velocity_->buffer_, 0, weights_grads_velocity_->buffer_count_ * sizeof(float));
    }

    backend_->device_malloc((void **)&loss_cache_, sizeof(float));

    LOG_INFO("GPT model created with use_grad=%d.", use_grad_);
}

GPT::~GPT()
{
    if (weights_)
    {
        delete weights_;
        weights_ = nullptr;
    }

    if (weights_grads_)
    {
        delete weights_grads_;
        weights_grads_ = nullptr;
    }

    if (activations_)
    {
        delete activations_;
        activations_ = nullptr;
    }

    if (cache_grads_)
    {
        delete cache_grads_;
        cache_grads_ = nullptr;
    }

    if (loss_cache_)
    {
        backend_->device_free(loss_cache_);
        loss_cache_ = nullptr;
    }

    if (weights_grads_momentum_)
    {
        delete weights_grads_momentum_;
        weights_grads_momentum_ = nullptr;
    }

    if (weights_grads_velocity_)
    {
        delete weights_grads_velocity_;
        weights_grads_velocity_ = nullptr;
    }

    LOG_INFO("GPT model destroyed.");
}

void GPT::init(float mean, float stddev)
{
    const int L = config_->num_layers;
    const int D = config_->d_model;
    const int F = config_->d_ffn;

    // Fill everything with normal distribution
    backend_->device_fill_normal(weights_->buffer_, mean, stddev, weights_->buffer_count_);

    // Residual projection scaling
    backend_->device_fill_normal(weights_->attn_proj_w, 0.0f, stddev / sqrtf(2 * L), L * D * D);
    backend_->device_fill_normal(weights_->ffn_down_w, 0.0f, stddev / sqrtf(2 * L), L * F * D);

    // LayerNorm weights to 1.0
    backend_->device_fill_const(weights_->ln_1_w, 1.0f, L * D);
    backend_->device_fill_const(weights_->ln_2_w, 1.0f, L * D);
    backend_->device_fill_const(weights_->ln_f_w, 1.0f, D);

    // Reset biases to zero
    backend_->device_memset(weights_->ln_1_b, 0, L * D * sizeof(float));
    backend_->device_memset(weights_->ln_2_b, 0, L * D * sizeof(float));
    backend_->device_memset(weights_->ln_f_b, 0, D * sizeof(float));
    backend_->device_memset(weights_->qkv_proj_b, 0, L * 3 * D * sizeof(float));
    backend_->device_memset(weights_->attn_proj_b, 0, L * D * sizeof(float));
    backend_->device_memset(weights_->ffn_up_b, 0, L * F * sizeof(float));
    backend_->device_memset(weights_->ffn_down_b, 0, L * D * sizeof(float));

    LOG_DEBUG("GPT weights initialized with mean=%.6f, stddev=%.6f.", mean, stddev);
}

bool GPT::load_checkpoint(const char *filename)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        LOG_ERROR("Could not open checkpoint file %s for reading.", filename);
        return false;
    }

    float *data = (float *)malloc(weights_->buffer_count_ * sizeof(float));

    if (!data)
    {
        LOG_ERROR("Could not allocate memory for checkpoint data.");
        fclose(fp);
        return false;
    }

    size_t count = fread(data, sizeof(float), weights_->buffer_count_, fp);
    fclose(fp);

    if (count != weights_->buffer_count_)
    {
        LOG_ERROR("Could not read entire checkpoint file %s.", filename);
        free(data);
        return false;
    }

    backend_->device_memcpy_h2d(weights_->buffer_, data, weights_->buffer_count_ * sizeof(float));
    free(data);

    LOG_INFO("Loaded checkpoint from %s successfully (%d floats, %.2f MB).", filename, (int)count, count * sizeof(float) / (1024.0 * 1024.0));
    return true;
}

bool GPT::save_checkpoint(const char *filename) const
{
    FILE *fp = fopen(filename, "wb");

    if (!fp)
    {
        LOG_ERROR("Could not open checkpoint file %s for writing.", filename);
        return false;
    }

    float *data = (float *)malloc(weights_->buffer_count_ * sizeof(float));

    if (!data)
    {
        LOG_ERROR("Could not allocate memory for checkpoint data.");
        fclose(fp);
        return false;
    }

    backend_->device_memcpy_d2h(data, weights_->buffer_, weights_->buffer_count_ * sizeof(float));

    size_t count = fwrite(data, sizeof(float), weights_->buffer_count_, fp);

    fclose(fp);
    free(data);

    if (count != weights_->buffer_count_)
    {
        LOG_ERROR("Could not write entire checkpoint file %s.", filename);
        return false;
    }

    LOG_INFO("Saved checkpoint to %s successfully (%d floats, %.2f MB).", filename, (int)count, count * sizeof(float) / (1024.0 * 1024.0));
    return true;
}

void GPT::set_size(int batch_size, int seq_len)
{
    if (activations_)
    {
        delete activations_;
        activations_ = nullptr;
    }

    if (cache_grads_)
    {
        delete cache_grads_;
        cache_grads_ = nullptr;
    }

    batch_size_ = batch_size;
    seq_len_ = seq_len;

    activations_ = new gpt_activations(backend_, config_, batch_size_, seq_len_);

    if (use_grad_)
    {
        cache_grads_ = new gpt_cache_x_grads(backend_, config_, batch_size_, seq_len_);
    }

    LOG_DEBUG("GPT model size set to batch_size=%d, seq_len=%d.", batch_size_, seq_len_);
}

void GPT::forward(const int *input_tokens)
{
    if (!activations_)
    {
        LOG_ERROR("activations not allocated. Call set_size() before forward().");
        return;
    }

    float *input = NULL;
    float *residual = NULL;

    // Embedding Layer
    backend_->device_embedding_forward(activations_->x_emb, weights_->wte, weights_->wpe, input_tokens, batch_size_, seq_len_, config_->d_model);
    input = activations_->x_emb;

    // Transformer Layers
    for (int layer = 0; layer < config_->num_layers; layer++)
    {
        // Calculate pointers for the current layer's weights and activations_
        float *ln_1_out = activations_->ln_1_out + layer * batch_size_ * seq_len_ * config_->d_model;
        float *ln_1_means = activations_->ln_1_means + layer * batch_size_ * seq_len_;
        float *ln_1_vars = activations_->ln_1_vars + layer * batch_size_ * seq_len_;
        float *ln_1_weights = weights_->ln_1_w + layer * config_->d_model;
        float *ln_1_bias = weights_->ln_1_b + layer * config_->d_model;

        float *qkv_out = activations_->qkv_out + layer * batch_size_ * seq_len_ * 3 * config_->d_model;
        float *qkv_w = weights_->qkv_proj_w + layer * config_->d_model * 3 * config_->d_model;
        float *qkv_b = weights_->qkv_proj_b + layer * 3 * config_->d_model;

        float *attn_logsumexp = activations_->attn_logsumexp + layer * batch_size_ * seq_len_ * config_->num_heads;
        float *attn_out = activations_->attn_out + layer * batch_size_ * seq_len_ * config_->d_model;

        float *attn_proj = activations_->attn_proj + layer * batch_size_ * seq_len_ * config_->d_model;
        float *attn_proj_w = weights_->attn_proj_w + layer * config_->d_model * config_->d_model;
        float *attn_proj_b = weights_->attn_proj_b + layer * config_->d_model;

        float *ln_2_out = activations_->ln_2_out + layer * batch_size_ * seq_len_ * config_->d_model;
        float *ln_2_means = activations_->ln_2_means + layer * batch_size_ * seq_len_;
        float *ln_2_vars = activations_->ln_2_vars + layer * batch_size_ * seq_len_;
        float *ln_2_weights = weights_->ln_2_w + layer * config_->d_model;
        float *ln_2_bias = weights_->ln_2_b + layer * config_->d_model;

        float *ffn_up = activations_->ffn_up + layer * batch_size_ * seq_len_ * config_->d_ffn;
        float *ffn_up_w = weights_->ffn_up_w + layer * config_->d_model * config_->d_ffn;
        float *ffn_up_b = weights_->ffn_up_b + layer * config_->d_ffn;
        float *ffn_act = activations_->ffn_act + layer * batch_size_ * seq_len_ * config_->d_ffn;
        float *ffn_down = activations_->ffn_down + layer * batch_size_ * seq_len_ * config_->d_model;
        float *ffn_down_w = weights_->ffn_down_w + layer * config_->d_ffn * config_->d_model;
        float *ffn_down_b = weights_->ffn_down_b + layer * config_->d_model;

        // LayerNorm 1 + Residual
        backend_->device_layernorm_residual_fused_forward(ln_1_out, ln_1_means, ln_1_vars, input, residual, ln_1_weights, ln_1_bias, batch_size_, seq_len_, config_->d_model);
        residual = input;
        input = ln_1_out;

        // QKV Projection
        backend_->device_linear_activation_fused_forward(qkv_out, nullptr, input, qkv_w, qkv_b, false, batch_size_, seq_len_, config_->d_model, 3 * config_->d_model);
        input = qkv_out;

        // Attention Mechanism
        backend_->device_flash_attention_forward(attn_out, attn_logsumexp, input, batch_size_, seq_len_, config_->d_model, config_->num_heads);
        input = attn_out;

        // Attention Projection
        backend_->device_linear_activation_fused_forward(attn_proj, nullptr, input, attn_proj_w, attn_proj_b, false, batch_size_, seq_len_, config_->d_model, config_->d_model);
        input = attn_proj;

        // LayerNorm 2 + Residual
        backend_->device_layernorm_residual_fused_forward(ln_2_out, ln_2_means, ln_2_vars, input, residual, ln_2_weights, ln_2_bias, batch_size_, seq_len_, config_->d_model);
        residual = input;
        input = ln_2_out;

        // Feedforward Network
        backend_->device_linear_activation_fused_forward(ffn_up, ffn_act, input, ffn_up_w, ffn_up_b, true, batch_size_, seq_len_, config_->d_model, config_->d_ffn);
        backend_->device_linear_activation_fused_forward(ffn_down, nullptr, ffn_act, ffn_down_w, ffn_down_b, false, batch_size_, seq_len_, config_->d_ffn, config_->d_model);
        input = ffn_down;
    }

    // LayerNorm Final + Residual
    backend_->device_layernorm_residual_fused_forward(activations_->ln_f_out, activations_->ln_f_means, activations_->ln_f_vars, input, residual, weights_->ln_f_w, weights_->ln_f_b, batch_size_, seq_len_, config_->d_model);
    input = activations_->ln_f_out;

    // Output Projection to logits
    backend_->device_unembedding_forward(activations_->logits, input, weights_->wte, batch_size_, seq_len_, config_->d_model, config_->vocab_size_padded);
    input = activations_->logits;

    // Softmax to probabilities (only if use_grad_ is true, otherwise skip to save computation)
    if (use_grad_)
    {
        backend_->device_softmax_forward(activations_->probs, input, batch_size_, seq_len_, config_->vocab_size, config_->vocab_size_padded);
    }
}

void GPT::backward(const int *input_tokens, const int *label_tokens)
{
    if (!use_grad_)
    {
        LOG_ERROR("Backward pass called but use_grad is false.");
        return;
    }

    if (!activations_ || !cache_grads_)
    {
        LOG_ERROR("activations or cache_x_grads not allocated. Call set_size() before backward().");
        return;
    }

    float *grad_y = NULL;
    float *grad_residual = NULL;

    // Cross Entropy + Softmax Backward
    backend_->device_cross_entropy_softmax_fused_backward(cache_grads_->softmax_final, activations_->probs, label_tokens, batch_size_, seq_len_, config_->vocab_size, config_->vocab_size_padded);
    grad_y = cache_grads_->softmax_final;

    // Unembedding Backward
    backend_->device_unembedding_backward(cache_grads_->unembedding, weights_grads_->wte, grad_y, activations_->ln_f_out, weights_->wte, batch_size_, seq_len_, config_->d_model, config_->vocab_size_padded);
    grad_y = cache_grads_->unembedding;

    // Final LayerNorm Backward
    float *last_layer_ffn_down = activations_->ffn_down + (config_->num_layers - 1) * batch_size_ * seq_len_ * config_->d_model;

    backend_->device_layernorm_residual_fused_backward(cache_grads_->ln_f, weights_grads_->ln_f_w, weights_grads_->ln_f_b, grad_y, nullptr, last_layer_ffn_down, activations_->ln_f_means, activations_->ln_f_vars, weights_->ln_f_w, batch_size_, seq_len_, config_->d_model);
    grad_y = cache_grads_->ln_f;

    // Backpropagation through Transformer layers
    for (int layer = config_->num_layers - 1; layer >= 0; layer--)
    {
        // Store the current grad_y for residual connection 2
        grad_residual = grad_y;

        // Backpropagation through Feedforward Network Down
        float *grad_x_ffn_down = cache_grads_->ffn_down + layer * batch_size_ * seq_len_ * config_->d_ffn;
        float *grad_w_ffn_down = weights_grads_->ffn_down_w + layer * config_->d_ffn * config_->d_model;
        float *grad_b_ffn_down = weights_grads_->ffn_down_b + layer * config_->d_model;
        float *ffn_down_w = weights_->ffn_down_w + layer * config_->d_ffn * config_->d_model;
        float *ffn_act = activations_->ffn_act + layer * batch_size_ * seq_len_ * config_->d_ffn;

        backend_->device_linear_backward(grad_x_ffn_down, grad_w_ffn_down, grad_b_ffn_down, grad_y, ffn_act, ffn_down_w, batch_size_, seq_len_, config_->d_ffn, config_->d_model);
        grad_y = grad_x_ffn_down;

        // Backpropagation through Feedforward Network Activation
        float *grad_x_ffn_act = cache_grads_->ffn_act + layer * batch_size_ * seq_len_ * config_->d_ffn;
        float *ffn_up = activations_->ffn_up + layer * batch_size_ * seq_len_ * config_->d_ffn;

        backend_->device_activation_backward(grad_x_ffn_act, grad_y, ffn_up, batch_size_, seq_len_, config_->d_ffn);
        grad_y = grad_x_ffn_act;

        // Backpropagation through Feedforward Network Up
        float *grad_x_ffn_up = cache_grads_->ffn_up + layer * batch_size_ * seq_len_ * config_->d_model;
        float *grad_w_ffn_up = weights_grads_->ffn_up_w + layer * config_->d_model * config_->d_ffn;
        float *grad_b_ffn_up = weights_grads_->ffn_up_b + layer * config_->d_ffn;
        float *ffn_up_w = weights_->ffn_up_w + layer * config_->d_model * config_->d_ffn;
        float *ln_2_out = activations_->ln_2_out + layer * batch_size_ * seq_len_ * config_->d_model;

        backend_->device_linear_backward(grad_x_ffn_up, grad_w_ffn_up, grad_b_ffn_up, grad_y, ln_2_out, ffn_up_w, batch_size_, seq_len_, config_->d_model, config_->d_ffn);
        grad_y = grad_x_ffn_up;

        // Backpropagation through LayerNorm 2 + Residual Connection 2
        float *grad_x_ln_2 = cache_grads_->ln_2 + layer * batch_size_ * seq_len_ * config_->d_model;
        float *grad_w_ln_2 = weights_grads_->ln_2_w + layer * config_->d_model;
        float *grad_b_ln_2 = weights_grads_->ln_2_b + layer * config_->d_model;
        float *ln_2_means = activations_->ln_2_means + layer * batch_size_ * seq_len_;
        float *ln_2_vars = activations_->ln_2_vars + layer * batch_size_ * seq_len_;
        float *ln_2_gamma = weights_->ln_2_w + layer * config_->d_model;
        float *attn_proj = activations_->attn_proj + layer * batch_size_ * seq_len_ * config_->d_model;

        backend_->device_layernorm_residual_fused_backward(grad_x_ln_2, grad_w_ln_2, grad_b_ln_2, grad_y, grad_residual, attn_proj, ln_2_means, ln_2_vars, ln_2_gamma, batch_size_, seq_len_, config_->d_model);
        grad_y = grad_x_ln_2;

        // Store the current grad_y for residual connection 1
        grad_residual = grad_y;

        // Backpropagation through Attention Projection
        float *grad_x_attn_proj = cache_grads_->attn_proj + layer * batch_size_ * seq_len_ * config_->d_model;
        float *grad_w_attn_proj = weights_grads_->attn_proj_w + layer * config_->d_model * config_->d_model;
        float *grad_b_attn_proj = weights_grads_->attn_proj_b + layer * config_->d_model;
        float *attn_proj_w = weights_->attn_proj_w + layer * config_->d_model * config_->d_model;
        float *attn_out = activations_->attn_out + layer * batch_size_ * seq_len_ * config_->d_model;

        backend_->device_linear_backward(grad_x_attn_proj, grad_w_attn_proj, grad_b_attn_proj, grad_y, attn_out, attn_proj_w, batch_size_, seq_len_, config_->d_model, config_->d_model);
        grad_y = grad_x_attn_proj;

        // Backpropagation through Attention Mechanism
        float *grad_x_attn = cache_grads_->attn + layer * batch_size_ * seq_len_ * 3 * config_->d_model;
        float *attn_d_helper = cache_grads_->attn_d_helper + layer * batch_size_ * seq_len_ * config_->num_heads;
        float *attn_logsumexp = activations_->attn_logsumexp + layer * batch_size_ * seq_len_ * config_->num_heads;
        float *qkv_out = activations_->qkv_out + layer * batch_size_ * seq_len_ * 3 * config_->d_model;

        backend_->device_flash_attention_backward(grad_x_attn, attn_d_helper, attn_logsumexp, grad_y, attn_out, qkv_out, batch_size_, seq_len_, config_->d_model, config_->num_heads);
        grad_y = grad_x_attn;

        // Backpropagation through QKV Projection
        float *grad_x_qkv = cache_grads_->qkv_proj + layer * batch_size_ * seq_len_ * config_->d_model;
        float *grad_w_qkv = weights_grads_->qkv_proj_w + layer * config_->d_model * 3 * config_->d_model;
        float *grad_b_qkv = weights_grads_->qkv_proj_b + layer * 3 * config_->d_model;
        float *qkv_proj_w = weights_->qkv_proj_w + layer * config_->d_model * 3 * config_->d_model;
        float *ln_1_out = activations_->ln_1_out + layer * batch_size_ * seq_len_ * config_->d_model;

        backend_->device_linear_backward(grad_x_qkv, grad_w_qkv, grad_b_qkv, grad_y, ln_1_out, qkv_proj_w, batch_size_, seq_len_, config_->d_model, 3 * config_->d_model);
        grad_y = grad_x_qkv;

        // Backpropagation through LayerNorm 1 + Residual Connection 1
        float *grad_x_ln_1 = cache_grads_->ln_1 + layer * batch_size_ * seq_len_ * config_->d_model;
        float *grad_w_ln_1 = weights_grads_->ln_1_w + layer * config_->d_model;
        float *grad_b_ln_1 = weights_grads_->ln_1_b + layer * config_->d_model;
        float *ln_1_means = activations_->ln_1_means + layer * batch_size_ * seq_len_;
        float *ln_1_vars = activations_->ln_1_vars + layer * batch_size_ * seq_len_;
        float *ln_1_gamma = weights_->ln_1_w + layer * config_->d_model;
        float *ln_1_input = layer == 0 ? activations_->x_emb : activations_->ffn_down + (layer - 1) * batch_size_ * seq_len_ * config_->d_model;

        backend_->device_layernorm_residual_fused_backward(grad_x_ln_1, grad_w_ln_1, grad_b_ln_1, grad_y, grad_residual, ln_1_input, ln_1_means, ln_1_vars, ln_1_gamma, batch_size_, seq_len_, config_->d_model);
        grad_y = grad_x_ln_1;
    }

    // Embedding Backward
    backend_->device_embedding_backward(weights_grads_->wte, weights_grads_->wpe, grad_y, input_tokens, batch_size_, seq_len_, config_->d_model);
}

void GPT::unscale_and_clip_grads(float max_norm, int accum_steps)
{
    backend_->device_scale_and_clip_grad(weights_grads_->buffer_, weights_grads_->buffer_count_, max_norm, accum_steps);
}

void GPT::optimizer_step(float learning_rate, float beta1, float beta2, float decay)
{
    if (!use_grad_)
    {
        LOG_ERROR("optimizer_step called but use_grad is false.");
        return;
    }

    current_step_++;

    const float eps = 1e-8f;

    backend_->device_adamw_step(weights_->wte, weights_grads_->wte, weights_grads_momentum_->wte, weights_grads_velocity_->wte, config_->vocab_size_padded * config_->d_model, learning_rate, beta1, beta2, eps, decay, current_step_);
    backend_->device_adamw_step(weights_->wpe, weights_grads_->wpe, weights_grads_momentum_->wpe, weights_grads_velocity_->wpe, config_->max_seq_len * config_->d_model, learning_rate, beta1, beta2, eps, decay, current_step_);
    backend_->device_adamw_step(weights_->ln_1_w, weights_grads_->ln_1_w, weights_grads_momentum_->ln_1_w, weights_grads_velocity_->ln_1_w, config_->num_layers * config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->ln_1_b, weights_grads_->ln_1_b, weights_grads_momentum_->ln_1_b, weights_grads_velocity_->ln_1_b, config_->num_layers * config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->qkv_proj_w, weights_grads_->qkv_proj_w, weights_grads_momentum_->qkv_proj_w, weights_grads_velocity_->qkv_proj_w, config_->num_layers * config_->d_model * 3 * config_->d_model, learning_rate, beta1, beta2, eps, decay, current_step_);
    backend_->device_adamw_step(weights_->qkv_proj_b, weights_grads_->qkv_proj_b, weights_grads_momentum_->qkv_proj_b, weights_grads_velocity_->qkv_proj_b, config_->num_layers * 3 * config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->attn_proj_w, weights_grads_->attn_proj_w, weights_grads_momentum_->attn_proj_w, weights_grads_velocity_->attn_proj_w, config_->num_layers * config_->d_model * config_->d_model, learning_rate, beta1, beta2, eps, decay, current_step_);
    backend_->device_adamw_step(weights_->attn_proj_b, weights_grads_->attn_proj_b, weights_grads_momentum_->attn_proj_b, weights_grads_velocity_->attn_proj_b, config_->num_layers * config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->ln_2_w, weights_grads_->ln_2_w, weights_grads_momentum_->ln_2_w, weights_grads_velocity_->ln_2_w, config_->num_layers * config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->ln_2_b, weights_grads_->ln_2_b, weights_grads_momentum_->ln_2_b, weights_grads_velocity_->ln_2_b, config_->num_layers * config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->ffn_up_w, weights_grads_->ffn_up_w, weights_grads_momentum_->ffn_up_w, weights_grads_velocity_->ffn_up_w, config_->num_layers * config_->d_ffn * config_->d_model, learning_rate, beta1, beta2, eps, decay, current_step_);
    backend_->device_adamw_step(weights_->ffn_up_b, weights_grads_->ffn_up_b, weights_grads_momentum_->ffn_up_b, weights_grads_velocity_->ffn_up_b, config_->num_layers * config_->d_ffn, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->ffn_down_w, weights_grads_->ffn_down_w, weights_grads_momentum_->ffn_down_w, weights_grads_velocity_->ffn_down_w, config_->num_layers * config_->d_model * config_->d_ffn, learning_rate, beta1, beta2, eps, decay, current_step_);
    backend_->device_adamw_step(weights_->ffn_down_b, weights_grads_->ffn_down_b, weights_grads_momentum_->ffn_down_b, weights_grads_velocity_->ffn_down_b, config_->num_layers * config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->ln_f_w, weights_grads_->ln_f_w, weights_grads_momentum_->ln_f_w, weights_grads_velocity_->ln_f_w, config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
    backend_->device_adamw_step(weights_->ln_f_b, weights_grads_->ln_f_b, weights_grads_momentum_->ln_f_b, weights_grads_velocity_->ln_f_b, config_->d_model, learning_rate, beta1, beta2, eps, 0.0f, current_step_);
}

void GPT::zero_grad()
{
    if (weights_grads_)
    {
        backend_->device_memset(weights_grads_->buffer_, 0, weights_grads_->buffer_count_ * sizeof(float));
    }

    if (cache_grads_)
    {
        backend_->device_memset(cache_grads_->buffer_, 0, cache_grads_->buffer_count_ * sizeof(float));
    }
}

float GPT::loss(const int *label_tokens)
{
    if (!use_grad_)
    {
        LOG_ERROR("loss() called but use_grad is false.");
        return -1.0f;
    }

    if (!activations_)
    {
        LOG_ERROR("activations not allocated. Call set_size() and forward() before loss().");
        return -1.0f;
    }

    backend_->device_cross_entropy_loss(loss_cache_, activations_->probs, label_tokens, batch_size_, seq_len_, config_->vocab_size_padded);

    float loss_value;
    backend_->device_memcpy_d2h(&loss_value, loss_cache_, sizeof(float));

    return loss_value;
}