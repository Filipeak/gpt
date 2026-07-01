#pragma once

struct Candidate
{
    int id;
    float logit;
    float prob;
};

class TokenSampler
{
public:
    TokenSampler(int vocab_size, float temperature, int top_k, float top_p);
    ~TokenSampler();

    int sample(float *new_logits);

private:
    Candidate *candidates_ = nullptr;
    int vocab_size_ = 0;
    float temperature_ = 0.0f;
    int top_k_ = 0;
    float top_p_ = 0.0f;
};