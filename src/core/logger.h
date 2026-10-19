#pragma once

#include <cstdio>

// Logs go to stderr, so stdout stays free for program output.
#ifndef NDEBUG
#define LOG_DEBUG(msg, ...) fprintf(stderr, "[debug] " msg "\n", ##__VA_ARGS__)
#else
#define LOG_DEBUG(msg, ...) ((void)0)
#endif

#define LOG_INFO(msg, ...) fprintf(stderr, "[info] " msg "\n", ##__VA_ARGS__)
#define LOG_WARN(msg, ...) fprintf(stderr, "[warn] " msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) fprintf(stderr, "[error] " msg "\n", ##__VA_ARGS__)
