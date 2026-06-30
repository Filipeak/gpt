#pragma once

float get_lr_cosine_decay(int step, int warmup, int total, float max_lr, float min_lr);