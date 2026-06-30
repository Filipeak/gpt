#include "cpu_backend.h"
#include "tensor_utils.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void CPUBackend::device_get_memory_info(float *free_mb, float *total_mb)
{
    *free_mb = FLT_MAX;
    *total_mb = FLT_MAX;
}

void CPUBackend::device_malloc(void **ptr, size_t size)
{
    *ptr = malloc(size);
}

void CPUBackend::device_free(void *ptr)
{
    free(ptr);
}

void CPUBackend::device_memset(void *ptr, int value, size_t size)
{
    memset(ptr, value, size);
}

void CPUBackend::device_fill_normal(float *ptr, float mean, float stddev, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        // Simple normal distribution sampling using Box-Muller transform
        float u1 = static_cast<float>(rand()) / RAND_MAX;
        float u2 = static_cast<float>(rand()) / RAND_MAX;
        float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);

        ptr[i] = z0 * stddev + mean;
    }
}

void CPUBackend::device_fill_const(float *ptr, float value, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        ptr[i] = value;
    }
}

void CPUBackend::device_memcpy_d2h(void *dst, const void *src, size_t size)
{
    memcpy(dst, src, size);
}

void CPUBackend::device_memcpy_h2d(void *dst, const void *src, size_t size)
{
    memcpy(dst, src, size);
}

void CPUBackend::device_embedding_forward(float *y, const float *wte, const float *wpe, const int *input_tokens, int batch_size, int seq_len, int hidden_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            int token_id = input_tokens[b * seq_len + s];

            for (int h = 0; h < hidden_size; ++h)
            {
                y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] = wte[TENSOR_IDX_2D(token_id, h, hidden_size)] + wpe[TENSOR_IDX_2D(s, h, hidden_size)];
            }
        }
    }
}

void CPUBackend::device_unembedding_forward(float *y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int v = 0; v < vocab_size; ++v)
            {
                float sum = 0.0f;

                for (int h = 0; h < hidden_size; ++h)
                {
                    sum += x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] * wte[TENSOR_IDX_2D(v, h, hidden_size)];
                }

                y[TENSOR_IDX_3D(b, s, v, seq_len, vocab_size)] = sum;
            }
        }
    }
}

void CPUBackend::device_embedding_backward(float *grad_wte, float *grad_wpe, const float *grad_y, const int *input_tokens, int batch_size, int seq_len, int hidden_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            int token_id = input_tokens[b * seq_len + s];

            for (int h = 0; h < hidden_size; ++h)
            {
                grad_wte[TENSOR_IDX_2D(token_id, h, hidden_size)] += grad_y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)];
                grad_wpe[TENSOR_IDX_2D(s, h, hidden_size)] += grad_y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)];
            }
        }
    }
}

void CPUBackend::device_unembedding_backward(float *grad_x, float *grad_wte, const float *grad_y, const float *x, const float *wte, int batch_size, int seq_len, int hidden_size, int vocab_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < hidden_size; ++h)
            {
                float sum = 0.0f;

                for (int v = 0; v < vocab_size; ++v)
                {
                    sum += grad_y[TENSOR_IDX_3D(b, s, v, seq_len, vocab_size)] * wte[TENSOR_IDX_2D(v, h, hidden_size)];

                    grad_wte[TENSOR_IDX_2D(v, h, hidden_size)] += grad_y[TENSOR_IDX_3D(b, s, v, seq_len, vocab_size)] * x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)];
                }

                grad_x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] = sum;
            }
        }
    }
}

void CPUBackend::device_layernorm_forward(float *y, float *means, float *vars, const float *x, const float *gamma, const float *beta, int batch_size, int seq_len, int hidden_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            float mean = 0.0f;
            float var = 0.0f;

            for (int h = 0; h < hidden_size; ++h)
            {
                mean += x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)];
            }
            mean /= hidden_size;

            for (int h = 0; h < hidden_size; ++h)
            {
                float diff = x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] - mean;
                var += diff * diff;
            }
            var /= hidden_size;

            means[b * seq_len + s] = mean;
            vars[b * seq_len + s] = var;

            for (int h = 0; h < hidden_size; ++h)
            {
                y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] = ((x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] - mean) / sqrtf(var + 1e-5f)) * gamma[h] + beta[h];
            }
        }
    }
}

void CPUBackend::device_layernorm_backward(float *grad_x, float *grad_gamma, float *grad_beta, const float *grad_y, const float *x, const float *means, const float *vars, const float *gamma, int batch_size, int seq_len, int hidden_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            float mean = means[TENSOR_IDX_2D(b, s, seq_len)];
            float var = vars[TENSOR_IDX_2D(b, s, seq_len)];

            float inv_std = 1.0f / sqrtf(var + 1e-5f);

            float sum1 = 0.0f;
            float sum2 = 0.0f;

            for (int h = 0; h < hidden_size; ++h)
            {
                float x_hat = (x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] - mean) * inv_std;

                grad_gamma[h] += grad_y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] * x_hat;
                grad_beta[h] += grad_y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)];

                sum1 += grad_y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] * gamma[h];
                sum2 += grad_y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] * gamma[h] * x_hat;
            }

            for (int h = 0; h < hidden_size; ++h)
            {
                float x_hat = (x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] - mean) * inv_std;

                grad_x[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] = inv_std * (grad_y[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] * gamma[h] - (sum1 / hidden_size) - x_hat * (sum2 / hidden_size));
            }
        }
    }
}

void CPUBackend::device_linear_activation_fused_forward(float *y, float *act, const float *x, const float *w, const float *b, bool activation, int batch_size, int seq_len, int input_size, int output_size)
{
    for (int bc = 0; bc < batch_size; ++bc)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int o = 0; o < output_size; ++o)
            {
                float sum = 0.0f;

                for (int i = 0; i < input_size; ++i)
                {
                    sum += x[TENSOR_IDX_3D(bc, s, i, seq_len, input_size)] * w[TENSOR_IDX_2D(i, o, output_size)];
                }

                sum += b[o];

                y[TENSOR_IDX_3D(bc, s, o, seq_len, output_size)] = sum;

                if (activation)
                {
                    act[TENSOR_IDX_3D(bc, s, o, seq_len, output_size)] = 0.5f * sum * (1.0f + tanhf(0.79788456f * (sum + 0.044715f * sum * sum * sum))); // GELU activation
                }
            }
        }
    }
}

void CPUBackend::device_linear_backward(float *grad_x, float *grad_w, float *grad_b, const float *grad_y, const float *x, const float *w, int batch_size, int seq_len, int input_size, int output_size)
{
    for (int bc = 0; bc < batch_size; ++bc)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int i = 0; i < input_size; ++i)
            {
                grad_x[TENSOR_IDX_3D(bc, s, i, seq_len, input_size)] = 0.0f;
            }

            for (int o = 0; o < output_size; ++o)
            {
                float grad_y_val = grad_y[TENSOR_IDX_3D(bc, s, o, seq_len, output_size)];

                for (int i = 0; i < input_size; ++i)
                {
                    grad_x[TENSOR_IDX_3D(bc, s, i, seq_len, input_size)] += grad_y_val * w[TENSOR_IDX_2D(i, o, output_size)];
                    grad_w[TENSOR_IDX_2D(i, o, output_size)] += x[TENSOR_IDX_3D(bc, s, i, seq_len, input_size)] * grad_y_val;
                }

                grad_b[o] += grad_y_val;
            }
        }
    }
}

void CPUBackend::device_activation_backward(float *grad_x, const float *grad_y, const float *x, int batch_size, int seq_len, int output_size)
{
    for (int bc = 0; bc < batch_size; ++bc)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int o = 0; o < output_size; ++o)
            {
                float x_val = x[TENSOR_IDX_3D(bc, s, o, seq_len, output_size)];
                float grad_y_val = grad_y[TENSOR_IDX_3D(bc, s, o, seq_len, output_size)];

                // Derivative of GELU activation
                float tanh_out = tanhf(0.79788456f * (x_val + 0.044715f * x_val * x_val * x_val));
                float gelu_derivative = 0.5f * (1.0f + tanh_out) + 0.5f * x_val * (1.0f - tanh_out * tanh_out) * (0.79788456f + 0.107032f * x_val * x_val);

                grad_x[TENSOR_IDX_3D(bc, s, o, seq_len, output_size)] = grad_y_val * gelu_derivative;
            }
        }
    }
}

void CPUBackend::device_attention_forward(float *y, float *scores, const float *qkv, int batch_size, int seq_len, int hidden_size, int num_heads)
{
    int stride = 3 * hidden_size;
    int head_dim = hidden_size / num_heads;
    float inv_dk = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < batch_size; ++b)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            const float *q_head = &qkv[TENSOR_IDX_3D(b, 0, 0 * hidden_size + h * head_dim, seq_len, stride)];
            const float *k_head = &qkv[TENSOR_IDX_3D(b, 0, 1 * hidden_size + h * head_dim, seq_len, stride)];
            const float *v_head = &qkv[TENSOR_IDX_3D(b, 0, 2 * hidden_size + h * head_dim, seq_len, stride)];
            float *scores_head = &scores[TENSOR_IDX_4D(b, h, 0, 0, num_heads, seq_len, seq_len)];

            for (int s1 = 0; s1 < seq_len; ++s1)
            {
                for (int s2 = 0; s2 < seq_len; ++s2)
                {
                    float score = 0.0f;

                    for (int d = 0; d < head_dim; ++d)
                    {
                        score += q_head[TENSOR_IDX_2D(s1, d, stride)] * k_head[TENSOR_IDX_2D(s2, d, stride)];
                    }

                    scores_head[TENSOR_IDX_2D(s1, s2, seq_len)] = score * inv_dk;
                }

                // Masking + Softmax over scores for each query position
                float max_score = -FLT_MAX;

                for (int s2 = 0; s2 < seq_len; ++s2)
                {
                    if (s2 > s1) // Mask future positions
                    {
                        continue;
                    }

                    float score = scores_head[TENSOR_IDX_2D(s1, s2, seq_len)];

                    if (score > max_score)
                    {
                        max_score = score;
                    }
                }

                float sum_exp = 0.0f;

                for (int s2 = 0; s2 < seq_len; ++s2)
                {
                    if (s2 > s1) // Mask future positions
                    {
                        continue;
                    }

                    float score = scores_head[TENSOR_IDX_2D(s1, s2, seq_len)];
                    float exp_score = expf(score - max_score);

                    scores_head[TENSOR_IDX_2D(s1, s2, seq_len)] = exp_score;
                    sum_exp += exp_score;
                }

                for (int s2 = 0; s2 < seq_len; ++s2)
                {
                    if (s2 > s1) // Mask future positions
                    {
                        scores_head[TENSOR_IDX_2D(s1, s2, seq_len)] = 0.0f;
                    }
                    else
                    {
                        scores_head[TENSOR_IDX_2D(s1, s2, seq_len)] /= sum_exp;
                    }
                }
            }

            // Compute output y = softmax(scores) @ V
            for (int s1 = 0; s1 < seq_len; ++s1)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    float sum = 0.0f;

                    for (int s2 = 0; s2 < seq_len; ++s2)
                    {
                        sum += scores_head[TENSOR_IDX_2D(s1, s2, seq_len)] * v_head[TENSOR_IDX_2D(s2, d, stride)];
                    }

                    y[TENSOR_IDX_3D(b, s1, h * head_dim + d, seq_len, hidden_size)] = sum;
                }
            }
        }
    }
}

void CPUBackend::device_attention_backward(float *grad_x, float *grad_softmax_cache, const float *grad_y, const float *qkv, const float *attn_scores, int batch_size, int seq_len, int hidden_size, int num_heads)
{
    int stride = 3 * hidden_size;
    int head_dim = hidden_size / num_heads;
    float inv_dk = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < batch_size; ++b)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            const float *q_head = &qkv[TENSOR_IDX_3D(b, 0, 0 * hidden_size + h * head_dim, seq_len, stride)];
            const float *k_head = &qkv[TENSOR_IDX_3D(b, 0, 1 * hidden_size + h * head_dim, seq_len, stride)];
            const float *v_head = &qkv[TENSOR_IDX_3D(b, 0, 2 * hidden_size + h * head_dim, seq_len, stride)];
            const float *scores_head = &attn_scores[TENSOR_IDX_4D(b, h, 0, 0, num_heads, seq_len, seq_len)];
            const float *grad_y_head = &grad_y[TENSOR_IDX_3D(b, 0, h * head_dim, seq_len, hidden_size)];
            float *grad_softmax_cache_head = &grad_softmax_cache[TENSOR_IDX_4D(b, h, 0, 0, num_heads, seq_len, seq_len)];

            // Compute grad_softmax_cache = grad_y @ V^T
            for (int s1 = 0; s1 < seq_len; ++s1)
            {
                for (int s2 = 0; s2 < seq_len; ++s2)
                {
                    float sum = 0.0f;

                    for (int d = 0; d < head_dim; ++d)
                    {
                        sum += grad_y_head[TENSOR_IDX_2D(s1, d, hidden_size)] * v_head[TENSOR_IDX_2D(s2, d, stride)];
                    }

                    grad_softmax_cache_head[TENSOR_IDX_2D(s1, s2, seq_len)] = sum;
                }
            }

            // Softmax backward
            for (int s1 = 0; s1 < seq_len; ++s1)
            {
                float dot_product = 0.0f;

                for (int s2 = 0; s2 < seq_len; ++s2)
                {
                    dot_product += grad_softmax_cache_head[TENSOR_IDX_2D(s1, s2, seq_len)] * scores_head[TENSOR_IDX_2D(s1, s2, seq_len)];
                }

                for (int s2 = 0; s2 < seq_len; ++s2)
                {
                    grad_softmax_cache_head[TENSOR_IDX_2D(s1, s2, seq_len)] = scores_head[TENSOR_IDX_2D(s1, s2, seq_len)] * (grad_softmax_cache_head[TENSOR_IDX_2D(s1, s2, seq_len)] - dot_product);
                }
            }

            // Compute grad_q = grad_softmax_cache @ K / sqrt(d_k)
            float *grad_q_head = &grad_x[TENSOR_IDX_3D(b, 0, 0 * hidden_size + h * head_dim, seq_len, stride)];

            for (int s1 = 0; s1 < seq_len; ++s1)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    float sum = 0.0f;

                    for (int s2 = 0; s2 < seq_len; ++s2)
                    {
                        sum += grad_softmax_cache_head[TENSOR_IDX_2D(s1, s2, seq_len)] * k_head[TENSOR_IDX_2D(s2, d, stride)];
                    }

                    grad_q_head[TENSOR_IDX_2D(s1, d, stride)] = sum * inv_dk;
                }
            }

            // Compute grad_k = grad_softmax_cache^T @ Q / sqrt(d_k)
            float *grad_k_head = &grad_x[TENSOR_IDX_3D(b, 0, 1 * hidden_size + h * head_dim, seq_len, stride)];

            for (int s2 = 0; s2 < seq_len; ++s2)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    float sum = 0.0f;

                    for (int s1 = 0; s1 < seq_len; ++s1)
                    {
                        sum += grad_softmax_cache_head[TENSOR_IDX_2D(s1, s2, seq_len)] * q_head[TENSOR_IDX_2D(s1, d, stride)];
                    }

                    grad_k_head[TENSOR_IDX_2D(s2, d, stride)] = sum * inv_dk;
                }
            }

            // Compute grad_v = scores^T @ grad_y
            float *grad_v_head = &grad_x[TENSOR_IDX_3D(b, 0, 2 * hidden_size + h * head_dim, seq_len, stride)];

            for (int s2 = 0; s2 < seq_len; ++s2)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    float sum = 0.0f;

                    for (int s1 = 0; s1 < seq_len; ++s1)
                    {
                        sum += scores_head[TENSOR_IDX_2D(s1, s2, seq_len)] * grad_y_head[TENSOR_IDX_2D(s1, d, hidden_size)];
                    }

                    grad_v_head[TENSOR_IDX_2D(s2, d, stride)] = sum;
                }
            }
        }
    }
}

void CPUBackend::device_residual_forward(float *current, const float *residual, int batch_size, int seq_len, int hidden_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < hidden_size; ++h)
            {
                current[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] += residual[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)];
            }
        }
    }
}

void CPUBackend::device_residual_backward(float *grad_current, const float *grad_residual, int batch_size, int seq_len, int hidden_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < hidden_size; ++h)
            {
                grad_current[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)] += grad_residual[TENSOR_IDX_3D(b, s, h, seq_len, hidden_size)];
            }
        }
    }
}

void CPUBackend::device_softmax_forward(float *y, const float *x, int batch_size, int seq_len, int size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            float max_val = -FLT_MAX;

            for (int i = 0; i < size; ++i)
            {
                float val = x[TENSOR_IDX_3D(b, s, i, seq_len, size)];

                if (val > max_val)
                {
                    max_val = val;
                }
            }

            float sum_exp = 0.0f;

            for (int i = 0; i < size; ++i)
            {
                float exp_val = expf(x[TENSOR_IDX_3D(b, s, i, seq_len, size)] - max_val);

                y[TENSOR_IDX_3D(b, s, i, seq_len, size)] = exp_val;
                sum_exp += exp_val;
            }

            for (int i = 0; i < size; ++i)
            {
                y[TENSOR_IDX_3D(b, s, i, seq_len, size)] /= sum_exp;
            }
        }
    }
}

void CPUBackend::device_softmax_backward(float *grad_x, const float *grad_y, const float *y, int batch_size, int seq_len, int size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            float dot_product = 0.0f;

            for (int i = 0; i < size; ++i)
            {
                dot_product += grad_y[TENSOR_IDX_3D(b, s, i, seq_len, size)] * y[TENSOR_IDX_3D(b, s, i, seq_len, size)];
            }

            for (int i = 0; i < size; ++i)
            {
                grad_x[TENSOR_IDX_3D(b, s, i, seq_len, size)] = y[TENSOR_IDX_3D(b, s, i, seq_len, size)] * (grad_y[TENSOR_IDX_3D(b, s, i, seq_len, size)] - dot_product);
            }
        }
    }
}

void CPUBackend::device_cross_entropy_softmax_fused_backward(float *grad_x, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            int label = tokens_labels[b * seq_len + s];

            for (int v = 0; v < vocab_size; ++v)
            {
                float softmax_val = y_softmax[TENSOR_IDX_3D(b, s, v, seq_len, vocab_size)];
                float one_hot_val = (v == label) ? 1.0f : 0.0f;

                grad_x[TENSOR_IDX_3D(b, s, v, seq_len, vocab_size)] = (softmax_val - one_hot_val) / (float)(batch_size * seq_len);
            }
        }
    }
}

void CPUBackend::device_cross_entropy_loss(float *loss, const float *y_softmax, const int *tokens_labels, int batch_size, int seq_len, int vocab_size)
{
    *loss = 0.0f;

    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            int label = tokens_labels[b * seq_len + s];
            float softmax_val = y_softmax[TENSOR_IDX_3D(b, s, label, seq_len, vocab_size)];

            *loss -= logf(softmax_val + 1e-8f); // Add a small epsilon to avoid log(0)
        }
    }

    *loss /= (float)(batch_size * seq_len);
}

void CPUBackend::device_adamw_step(float *params, float *g, float *m, float *v, int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step)
{
    float beta1_pow = powf(beta1, step);
    float beta2_pow = powf(beta2, step);

    for (int i = 0; i < size; ++i)
    {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];

        float m_hat = m[i] / (1.0f - beta1_pow);
        float v_hat = v[i] / (1.0f - beta2_pow);

        params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params[i]);
    }
}

void CPUBackend::device_clip_grad_norm(float *g, int size, float max_norm)
{
    float norm = 0.0f;

    for (int i = 0; i < size; ++i)
    {
        norm += g[i] * g[i];
    }

    norm = sqrtf(norm);

    if (norm > max_norm)
    {
        float scale = max_norm / norm;

        for (int i = 0; i < size; ++i)
        {
            g[i] *= scale;
        }
    }
}