#include "lr_scheduling.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float get_lr_cosine_decay(int step, int warmup, int total, float max_lr, float min_lr)
{
    if (step < warmup) // linear warmup
    {
        return max_lr * (step + 1) / warmup;
    }

    if (step >= total)
    {
        return min_lr;
    }

    float prog = (float)(step - warmup) / (total - warmup); // 0..1

    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + cosf(M_PI * prog));
}