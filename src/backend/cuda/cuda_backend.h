#pragma once

#include "../backend_interface.h"

struct cublasContext;
struct curandGenerator_st;

class CUDABackend : public IGPTBackend
{
public:
    CUDABackend(unsigned long seed = 42);
    ~CUDABackend();

    void device_get_memory_info(float *free_mb, float *total_mb) override;
    void device_malloc(void **ptr, size_t size) override;
    void device_free(void *ptr) override;
    void device_memset(void *ptr, int value, size_t size) override;
    void device_fill_normal(float *ptr, float mean, float stddev, size_t size) override;
    void device_fill_const(float *ptr, float value, size_t size) override;
    void device_memcpy_d2h(void *dst, const void *src, size_t size) override;
    void device_memcpy_h2d(void *dst, const void *src, size_t size) override;
    void device_embedding_forward(float *y, const float *wte, const float *wpe, const int *input_tokens, int batch_size, int seq_len, int hidden_size) override;
    void device_unembedding_forward(float *y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size_padded) override;
    void device_embedding_backward(float *grad_wte, float *grad_wpe, const float *grad_y, const int *input_tokens, int batch_size, int seq_len, int hidden_size) override;
    void device_unembedding_backward(float *grad_x, float *grad_wte, const float *grad_y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size_padded) override;
    void device_layernorm_forward(float *y, float *means, float *vars, const float *x, const float *gamma, const float *beta, int batch_size, int seq_len, int hidden_size) override;
    void device_layernorm_backward(float *grad_x, float *grad_gamma, float *grad_beta, const float *grad_y, const float *x, const float *means, const float *vars, const float *gamma, int batch_size, int seq_len, int hidden_size) override;
    void device_linear_activation_fused_forward(float *y, float *act, const float *x, const float *w, const float *b, bool activation, int batch_size, int seq_len, int input_size, int output_size) override;
    void device_linear_backward(float *grad_x, float *grad_w, float *grad_b, const float *grad_y, const float *x, const float *w, int batch_size, int seq_len, int input_size, int output_size) override;
    void device_activation_backward(float *grad_x, const float *grad_y, const float *x, int batch_size, int seq_len, int output_size) override;
    void device_attention_forward(float *y, float *scores, const float *qkv, int batch_size, int seq_len, int hidden_size, int num_heads) override;
    void device_attention_backward(float *grad_x, float *grad_softmax_cache, const float *grad_y, const float *qkv, const float *attn_scores, int batch_size, int seq_len, int hidden_size, int num_heads) override;
    void device_residual_forward(float *current, const float *residual, int batch_size, int seq_len, int hidden_size) override;
    void device_residual_backward(float *grad_current, const float *grad_residual, int batch_size, int seq_len, int hidden_size) override;
    void device_softmax_forward(float *y, const float *x, int batch_size, int seq_len, int vocab_size, int vocab_size_padded) override;
    void device_cross_entropy_softmax_fused_backward(float *grad_x, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size, int vocab_size_padded) override;
    void device_cross_entropy_loss(float *loss, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size_padded) override;
    void device_adamw_step(float *params, float *g, float *m, float *v, int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step) override;
    void device_clip_grad_norm(float *g, int size, float max_norm) override;

private:
    cublasContext *cublas_handle_;
    curandGenerator_st *curand_handle_;
    float *clip_norm_device_;
};