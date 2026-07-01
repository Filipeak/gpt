#pragma once

#include "backend_interface.h"
#include <cstddef>

// Model configuration
struct gpt_config
{
    int max_seq_len;       // Maximum number of tokens in a sequence
    int vocab_size;        // Size of the vocabulary (number of unique tokens)
    int vocab_size_padded; // Size of the vocabulary padded to the nearest multiple of 128
    int num_layers;        // Number of Transformer layers
    int num_heads;         // Number of attention heads in each Transformer layer
    int d_model;           // Dimensionality of the model (hidden size)
    int d_ffn;             // Dimensionality of the feedforward network in each Transformer layer

    // Utility functions
    void print() const;
};

#define GPT_WEIGHTS_PARAMS_COUNT 16

// Pointers to model weights
struct gpt_weights
{
    // Input Embeddings
    float *wte; // Token Embeddings [vocab_size, d_model]
    float *wpe; // Positional Embeddings [max_seq_len, d_model]

    // Transformer layers
    float *ln_1_w; // LayerNorm 1 weights [num_layers, d_model]
    float *ln_1_b; // LayerNorm 1 bias [num_layers, d_model]

    float *qkv_proj_w; // QKV weights [num_layers, d_model, 3 * d_model]
    float *qkv_proj_b; // QKV bias [num_layers, 3 * d_model]

    float *attn_proj_w; // Projection weights [num_layers, d_model, d_model]
    float *attn_proj_b; // Projection bias [num_layers, d_model]

    float *ln_2_w; // LayerNorm 2 weights [num_layers, d_model]
    float *ln_2_b; // LayerNorm 2 bias [num_layers, d_model]

    float *ffn_up_w;   // Feedforward up weights [num_layers, d_model, d_ffn]
    float *ffn_up_b;   // Feedforward up bias [num_layers, d_ffn]
    float *ffn_down_w; // Feedforward down weights [num_layers, d_ffn, d_model]
    float *ffn_down_b; // Feedforward down bias [num_layers, d_model]

    // Final LayerNorm
    float *ln_f_w; // Final LayerNorm weights [d_model]
    float *ln_f_b; // Final LayerNorm bias [d_model]

    // Utility for memory management
    IGPTBackend *backend_;
    float *buffer_;
    size_t buffer_count_;

    gpt_weights(IGPTBackend *backend, const gpt_config *config);
    ~gpt_weights();
};

#define GPT_ACTIVATIONS_COUNT 19

// Activation Cache for the current Batch (B) and Sequence Length (T)
struct gpt_activations
{
    // Input
    float *x_emb; // Combined embeddings [B, T, d_model]

    // Transformer layers
    float *ln_1_out;   // LayerNorm 1 outputs [num_layers, B, T, d_model]
    float *ln_1_means; // LayerNorm 1 means [num_layers, B, T]
    float *ln_1_vars;  // LayerNorm 1 variances [num_layers, B, T]

    float *qkv_out;     // QKV concatenated [num_layers, B, T, 3 * d_model]
    float *attn_scores; // Attention scores [num_layers, B, num_heads, T, T]
    float *attn_out;    // Attention concatenated outputs [num_layers, B, T, d_model]
    float *attn_proj;   // Attention projection outputs [num_layers, B, T, d_model]

    float *ln_2_out;   // LayerNorm 2 outputs [num_layers, B, T, d_model]
    float *ln_2_means; // LayerNorm 2 means [num_layers, B, T]
    float *ln_2_vars;  // LayerNorm 2 variances [num_layers, B, T]

    float *ffn_up;   // Feedforward up outputs [num_layers, B, T, d_ffn]
    float *ffn_act;  // Feedforward activation outputs [num_layers, B, T, d_ffn]
    float *ffn_down; // Feedforward down outputs [num_layers, B, T, d_model]

    // Final
    float *ln_f_out;   // Final LayerNorm outputs [B, T, d_model]
    float *ln_f_means; // Final LayerNorm means [B, T]
    float *ln_f_vars;  // Final LayerNorm variances [B, T]
    float *logits;     // Output logits (unembedded) [B, T, vocab_size]
    float *probs;      // Output probabilities [B, T, vocab_size]

    // Utility for memory management
    IGPTBackend *backend_;
    float *buffer_;
    size_t buffer_count_;

    gpt_activations(IGPTBackend *backend, const gpt_config *config, int batch_size, int seq_len);
    ~gpt_activations();
};

#define GPT_CACHE_X_GRADS_COUNT 12

// Gradients for the backward pass
struct gpt_cache_x_grads
{
    // Transformer layers
    float *ln_1;         // [num_layers, B, T, d_model]
    float *qkv_proj;     // [num_layers, B, T, d_model]
    float *attn;         // [num_layers, B, T, 3 * d_model]
    float *attn_softmax; // [num_layers, B, num_heads, T, T]
    float *attn_proj;    // [num_layers, B, T, d_model]
    float *ln_2;         // [num_layers, B, T, d_model]
    float *ffn_up;       // [num_layers, B, T, d_model]
    float *ffn_act;      // [num_layers, B, T, d_ffn]
    float *ffn_down;     // [num_layers, B, T, d_ffn]

    // Final
    float *ln_f;          // [B, T, d_model]
    float *unembedding;   // [B, T, d_model]
    float *softmax_final; // [B, T, vocab_size]

    // Utility for memory management
    IGPTBackend *backend_;
    float *buffer_;
    size_t buffer_count_;

    gpt_cache_x_grads(IGPTBackend *backend, const gpt_config *config, int batch_size, int seq_len);
    ~gpt_cache_x_grads();
};

// Main GPT class
class GPT
{
public:
    GPT(IGPTBackend *backend, const gpt_config *config, bool use_grad);
    ~GPT();

    void init(float mean, float stddev);
    bool load_checkpoint(const char *filename); // TODO: Load also model config, not only binary weights
    bool save_checkpoint(const char *filename) const;

    void set_size(int batch_size, int seq_len);

    void forward(const int *input_tokens);
    void backward(const int *input_tokens, const int *label_tokens);
    void clip_grad_norm(float max_norm);
    void optimizer_step(float learning_rate, float beta1, float beta2, float decay);
    void zero_grad();
    float loss(const int *label_tokens);

    const gpt_weights *grad_weights() const { return weights_grads_; }
    const gpt_activations *activations() const { return activations_; }

private:
    IGPTBackend *backend_ = nullptr;
    const gpt_config *config_ = nullptr;

    bool use_grad_ = false;
    int batch_size_ = 0;
    int seq_len_ = 0;
    int current_step_ = 0;

    float *loss_cache_ = nullptr;
    gpt_weights *weights_ = nullptr;
    gpt_activations *activations_ = nullptr;
    gpt_weights *weights_grads_ = nullptr;
    gpt_cache_x_grads *cache_grads_ = nullptr;
    gpt_weights *weights_grads_momentum_ = nullptr;
    gpt_weights *weights_grads_velocity_ = nullptr;
};