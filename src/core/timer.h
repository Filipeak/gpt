#pragma once

#include <chrono>

// Host wall-clock timer. GPU work runs asynchronously, so when the measured span
// includes device work, call backend->device_synchronize() before elapsed_ms().
class Timer
{
public:
    Timer() { reset(); }

    void reset() { start_ = std::chrono::steady_clock::now(); }

    float elapsed_ms() const
    {
        return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};
