#pragma once

#include <cstdio>

#ifndef NDEBUG
#define LOG_DEBUG(msg, ...) printf("[debug] " msg "\n", ##__VA_ARGS__)
#else
#define LOG_DEBUG(msg, ...) ((void)0)
#endif

#define LOG_INFO(msg, ...) printf("[info] " msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) printf("[error] " msg "\n", ##__VA_ARGS__)