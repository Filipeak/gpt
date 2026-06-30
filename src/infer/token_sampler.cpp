#include "token_sampler.h"
#include <cstdlib>
#include <cmath>

struct Candidate
{
    int id;
    float logit;
    float prob;
};

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

int sample_token(float *logits, int vocab_size, float temperature, int top_k, float top_p)
{
    Candidate *cand = (Candidate *)malloc(sizeof(Candidate) * vocab_size);

    // Apply temperature scaling
    for (int i = 0; i < vocab_size; i++)
    {
        cand[i].id = i;
        cand[i].logit = logits[i] / temperature;
    }

    // Sort candidates by logit in descending order
    qsort(cand, vocab_size, sizeof(Candidate), compare_desc);

    if (top_k > 0 && top_k < vocab_size)
    {
        vocab_size = top_k;
    }

    // Softmax
    float max_logit = cand[0].logit;
    float sum_exp = 0.0f;

    for (int i = 0; i < vocab_size; i++)
    {
        cand[i].prob = expf(cand[i].logit - max_logit);
        sum_exp += cand[i].prob;
    }

    for (int i = 0; i < vocab_size; i++)
    {
        cand[i].prob /= sum_exp;
    }

    // Apply top-p filtering
    if (top_p > 0.0f && top_p < 1.0f)
    {
        float cumulative_prob = 0.0f;
        int new_vocab_size = 0;

        for (int i = 0; i < vocab_size; i++)
        {
            cumulative_prob += cand[i].prob;
            new_vocab_size++;

            if (cumulative_prob >= top_p)
            {
                break;
            }
        }

        vocab_size = new_vocab_size;

        float sum = 0.0f;

        for (int i = 0; i < vocab_size; i++)
        {
            sum += cand[i].prob;
        }

        for (int i = 0; i < vocab_size; i++)
        {
            cand[i].prob /= sum;
        }
    }

    // Sample a token based on the filtered probabilities
    float r = (float)(rand()) / (float)RAND_MAX;
    float cumulative = 0.0f;
    int token = cand[vocab_size - 1].id; // Default to the last token in case of rounding errors

    for (int i = 0; i < vocab_size; i++)
    {
        cumulative += cand[i].prob;

        if (r <= cumulative)
        {
            token = cand[i].id;
            break;
        }
    }

    // Free the allocated memory for candidates and return the sampled token
    free(cand);

    return token;
}