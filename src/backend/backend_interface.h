#pragma once

#include <cstddef>

class IGPTBackend
{
public:
    virtual ~IGPTBackend() = default;

    virtual void device_get_memory_info(float *free_mb, float *total_mb) = 0;

    virtual void device_malloc(void **ptr, size_t size) = 0;

    virtual void device_free(void *ptr) = 0;

    virtual void device_memset(void *ptr, int value, size_t size) = 0;

    virtual void device_fill_normal(float *ptr, float mean, float stddev, size_t size) = 0;

    virtual void device_fill_const(float *ptr, float value, size_t size) = 0;

    virtual void device_memcpy_d2h(void *dst, const void *src, size_t size) = 0;

    virtual void device_memcpy_h2d(void *dst, const void *src, size_t size) = 0;

    /**
     * Tensors:
     * - y              [batch_size, seq_len, hidden_size]
     * - wte            [vocab_size_padded, hidden_size]
     * - wpe            [seq_len, hidden_size]
     * - input_tokens   [batch_size, seq_len]
     *
     * Performs (with lookup tables):
     *  - y = x @ wte + p @ wpe
     *
     * Where:
     * - x = one-hot encoding of input tokens [batch_size, seq_len, vocab_size_padded]
     * - p = one-hot encoding of position indices [batch_size, seq_len, seq_len]
     */
    virtual void device_embedding_forward(float *y, const float *wte, const float *wpe, const int *input_tokens, int batch_size, int seq_len, int hidden_size) = 0;

    /**
     * Tensors:
     * - y              [batch_size, seq_len, vocab_size_padded]
     * - x              [batch_size, seq_len, hidden_size]
     * - wte            [vocab_size_padded, hidden_size]
     *
     * Performs:
     * - y = x @ wte^T
     */
    virtual void device_unembedding_forward(float *y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size) = 0;

    /**
     * Tensors:
     * - grad_wte       [vocab_size_padded, hidden_size]
     * - grad_wpe       [seq_len, hidden_size]
     * - grad_y         [batch_size, seq_len, hidden_size]
     * - input_tokens   [batch_size, seq_len]
     *
     * Performs:
     * - grad_wte += x^T @ grad_y
     * - grad_wpe += p^T @ grad_y
     *
     * Where:
     * - x = one-hot encoding of input tokens [batch_size, seq_len, vocab_size_padded]
     * - p = one-hot encoding of position indices [batch_size, seq_len, seq_len]
     */
    virtual void device_embedding_backward(float *grad_wte, float *grad_wpe, const float *grad_y, const int *input_tokens, int batch_size, int seq_len, int hidden_size) = 0;

    /**
     * Tensors:
     * - grad_x         [batch_size, seq_len, hidden_size]
     * - grad_wte       [vocab_size_padded, hidden_size]
     * - grad_y         [batch_size, seq_len, vocab_size_padded]
     * - x              [batch_size, seq_len, hidden_size]
     * - wte            [vocab_size_padded, hidden_size]
     *
     * Performs:
     * - grad_x = grad_y @ wte
     * - grad_wte += grad_y^T @ x
     */
    virtual void device_unembedding_backward(float *grad_x, float *grad_wte, const float *grad_y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size_padded) = 0;

    /**
     * Tensors:
     * - y              [batch_size, seq_len, hidden_size]
     * - means          [batch_size, seq_len]
     * - vars           [batch_size, seq_len]
     * - x              [batch_size, seq_len, hidden_size]
     * - residual       [batch_size, seq_len, hidden_size]
     * - gamma          [hidden_size]
     * - beta           [hidden_size]
     *
     * Performs:
     * - means = mean(x, axis=-1)
     * - vars = var(x, axis=-1)
     * - y = (x - means) / sqrt(vars + eps) * gamma + beta
     */
    virtual void device_layernorm_residual_fused_forward(float *y, float *means, float *vars, float *x, const float *residual, const float *gamma, const float *beta, int batch_size, int seq_len, int hidden_size) = 0;

    /**
     * Tensors:
     * - grad_x         [batch_size, seq_len, hidden_size]
     * - grad_gamma     [hidden_size]
     * - grad_beta      [hidden_size]
     * - grad_y         [batch_size, seq_len, hidden_size]
     * - grad_residual  [batch_size, seq_len, hidden_size]
     * - x              [batch_size, seq_len, hidden_size]
     * - means          [batch_size, seq_len]
     * - vars           [batch_size, seq_len]
     * - gamma          [hidden_size]
     *
     * Performs:
     * - grad_x = layernorm_backward(grad_y, x, means, vars, gamma)
     * - grad_gamma += sum(grad_y * (x - means) / sqrt(vars + eps), axis=0)
     * - grad_beta += sum(grad_y, axis=0)
     */
    virtual void device_layernorm_residual_fused_backward(float *grad_x, float *grad_gamma, float *grad_beta, const float *grad_y, const float *grad_residual, const float *x, const float *means, const float *vars, const float *gamma, int batch_size, int seq_len, int hidden_size) = 0;

    /**
     * Tensors:
     * - y              [batch_size, seq_len, output_size]
     * - act            [batch_size, seq_len, output_size]
     * - x              [batch_size, seq_len, input_size]
     * - w              [input_size, output_size]
     * - b              [output_size]
     *
     * Performs:
     * - y = x @ w + b
     * - act = activation(y) if activation is true, not used otherwise
     */
    virtual void device_linear_activation_fused_forward(float *y, float *act, const float *x, const float *w, const float *b, bool activation, int batch_size, int seq_len, int input_size, int output_size) = 0;

    /**
     * Tensors:
     * - grad_x         [batch_size, seq_len, input_size]
     * - grad_w         [input_size, output_size]
     * - grad_b         [output_size]
     * - grad_y         [batch_size, seq_len, output_size]
     * - x              [batch_size, seq_len, input_size]
     * - w              [input_size, output_size]
     *
     * Performs:
     * - grad_x = grad_y @ w^T
     * - grad_w += x^T @ grad_y
     * - grad_b += sum(grad_y, axis=0)
     */
    virtual void device_linear_backward(float *grad_x, float *grad_w, float *grad_b, const float *grad_y, const float *x, const float *w, int batch_size, int seq_len, int input_size, int output_size) = 0;

    /**
     * Tensors:
     * - grad_x         [batch_size, seq_len, output_size]
     * - grad_y         [batch_size, seq_len, output_size]
     * - x              [batch_size, seq_len, output_size]
     *
     * Performs:
     * - grad_x = grad_y * f'(x)
     */
    virtual void device_activation_backward(float *grad_x, const float *grad_y, const float *x, int batch_size, int seq_len, int output_size) = 0;

    /**
     * Tensors:
     * - y              [batch_size, seq_len, hidden_size]
     * - logsumexp      [batch_size, seq_len, num_heads]
     * - qkv            [batch_size, seq_len, 3 * hidden_size]
     *
     * Performs:
     * - y = FlashAttention Forward
     */
    virtual void device_flash_attention_forward(float *y, float *logsumexp, const float *qkv, int batch_size, int seq_len, int hidden_size, int num_heads) = 0;

    /**
     * Tensors:
     * - grad_x             [batch_size, seq_len, 3 * hidden_size]
     * - D_cache            [batch_size, seq_len, num_heads]
     * - logsumexp          [batch_size, seq_len, num_heads]
     * - grad_y             [batch_size, seq_len, hidden_size]
     * - y                  [batch_size, seq_len, hidden_size]
     * - qkv                [batch_size, seq_len, 3 * hidden_size]
     *
     * Performs:
     * - grad_x = FlashAttention Backward
     */
    virtual void device_flash_attention_backward(float *grad_x, float *D_cache, const float *logsumexp, const float *grad_y, const float *y, const float *qkv, int batch_size, int seq_len, int hidden_size, int num_heads) = 0;

    /**
     * Tensors:
     * - y              [batch_size, seq_len, vocab_size_padded]
     * - x              [batch_size, seq_len, vocab_size_padded]
     *
     * Performs:
     * - y = softmax(x, axis=-1)
     */
    virtual void device_softmax_forward(float *y, const float *x, int batch_size, int seq_len, int vocab_size, int vocab_size_padded) = 0;

    /**
     * Tensors:
     * - grad_x         [batch_size, seq_len, vocab_size_padded]
     * - y_softmax      [batch_size, seq_len, vocab_size_padded]
     * - tokens_labels  [batch_size, seq_len]
     *
     * Performs:
     * - grad_x = (y_softmax - one_hot(tokens_labels)) / (batch_size * seq_len)
     */
    virtual void device_cross_entropy_softmax_fused_backward(float *grad_x, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size, int vocab_size_padded) = 0;

    /**
     * Tensors:
     * - loss           [1]
     * - y_softmax      [batch_size, seq_len, vocab_size_padded]
     * - tokens_labels  [batch_size, seq_len]
     *
     * Performs:
     * - Mean cross-entropy loss computation
     */
    virtual void device_cross_entropy_loss(float *loss, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size_padded) = 0;

    /**
     * Tensors:
     * - params         [size]
     * - g              [size]
     * - m              [size]
     * - v              [size]
     *
     * Performs:
     * - AdamW parameter update step
     */
    virtual void device_adamw_step(float *params, float *g, float *m, float *v, int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step) = 0;

    /**
     * Tensors:
     * - g              [size]
     *
     * Performs:
     * - Gradient clipping by norm + Scaling for gradient accumulation
     */
    virtual void device_scale_and_clip_grad(float *g, int size, float max_norm, int accum_steps) = 0;
};