#pragma once

#include "backend_interface.h"

// Named profiler range that closes at the end of the scope:
//     ProfileRange range(backend, "forward");
class ProfileRange
{
public:
    ProfileRange(IGPTBackend *backend, const char *name) : backend_(backend) { backend_->profile_range_push(name); }
    ~ProfileRange() { backend_->profile_range_pop(); }

    ProfileRange(const ProfileRange &) = delete;
    ProfileRange &operator=(const ProfileRange &) = delete;

private:
    IGPTBackend *backend_;
};
