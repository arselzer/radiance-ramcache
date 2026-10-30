/* tests/r4d_emu: the part of HIP's hip_bf16.h the bf16 GEMM units use. __hip_bfloat16 holds the
 * bf16 bits and is made from a float by round to nearest even with a NaN kept quiet, as HIP's
 * conversion (the language's float to __bf16) does. Test-only. */
#pragma once
#include <cstdint>
#include <cstring>

struct __hip_bfloat16 {
    unsigned short __x;
    __hip_bfloat16() = default;
    __hip_bfloat16(float v) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        if ((u & 0x7fffffffu) > 0x7f800000u) __x = (unsigned short)((u >> 16) | 0x0040u);
        else __x = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
};
static_assert(sizeof(__hip_bfloat16) == 2, "a bf16 is two bytes");
