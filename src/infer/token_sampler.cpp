#include "token_sampler.h"
#include <cstdlib>
#include <cmath>

static int compare_desc(const void *a, const void *b)
{
    const Candidate *x = (const Candidate *)a;
    const Candidate *y = (const Candidate *)b;

    if (x->logit < y->logit)
    {
        return 1;
    }
    else if (x->logit > y->logit)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

TokenSampler::TokenSampler(int vocab_size, float temperature, int top_k, float top_p) : vocab_size_(vocab_size), temperature_(temperature), top_k_(top_k), top_p_(top_p)
{
    candidates_ = (Candidate *)malloc(sizeof(Candidate) * vocab_size_);
}

TokenSampler::~TokenSampler()
{
    free(candidates_);
}

int TokenSampler::sample(float *new_logits)
{
    // Apply temperature scaling
    for (int i = 0; i < vocab_size_; i++)
    {
        candidates_[i].id = i;
        candidates_[i].logit = new_logits[i] / temperature_;
    }

    // Sort candidates by logit in descending order
    qsort(candidates_, vocab_size_, sizeof(Candidate), compare_desc);

    int temp_size = vocab_size_;

    if (top_k_ > 0 && top_k_ < vocab_size_)
    {
        temp_size = top_k_;
    }

    // Softmax
    float max_logit = candidates_[0].logit;
    float sum_exp = 0.0f;

    for (int i = 0; i < temp_size; i++)
    {
        candidates_[i].prob = expf(candidates_[i].logit - max_logit);
        sum_exp += candidates_[i].prob;
    }

    for (int i = 0; i < temp_size; i++)
    {
        candidates_[i].prob /= sum_exp;
    }

    // Apply top-p filtering
    if (top_p_ > 0.0f && top_p_ < 1.0f)
    {
        float cumulative_prob = 0.0f;
        int new_vocab_size = 0;

        for (int i = 0; i < temp_size; i++)
        {
            cumulative_prob += candidates_[i].prob;
            new_vocab_size++;

            if (cumulative_prob >= top_p_)
            {
                break;
            }
        }

        temp_size = new_vocab_size;

        float sum = 0.0f;

        for (int i = 0; i < temp_size; i++)
        {
            sum += candidates_[i].prob;
        }

        for (int i = 0; i < temp_size; i++)
        {
            candidates_[i].prob /= sum;
        }
    }

    // Sample a token based on the filtered probabilities
    float r = (float)(rand()) / (float)RAND_MAX;
    float cumulative = 0.0f;
    int token = candidates_[temp_size - 1].id; // Default to the last token in case of rounding errors

    for (int i = 0; i < temp_size; i++)
    {
        cumulative += candidates_[i].prob;

        if (r <= cumulative)
        {
            token = candidates_[i].id;
            break;
        }
    }

    return token;
}