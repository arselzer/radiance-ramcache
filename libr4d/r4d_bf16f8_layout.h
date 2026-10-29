/* r4d_bf16f8_layout.h -- the stored form of gemm_nt_bf16f8's weight: a bf16 [N][K] weight every
 * 32x32 block of which is E4M3 codes times one power of two, stored as those codes and one E8M0
 * exponent a block. r4d_layout.cpp's bf16f8 hooks write it from the bf16 plane and refuse a weight
 * that is not exactly so; r4d_gemm_bf16f8_nt_m64.hip expands it in registers to the bf16 bits
 * gemm_bf16_nt_m64 reads from the plane. Pure C, so the host hooks and the kernel include it alike.
 *
 * For this rank's weight, N % 16 == 0 and K % 32 == 0:
 *
 *   codes   bytes [0, N*K), 512-byte packets. Packet t*(K/32) + kb holds 16-row tile t and 32-K
 *           block kb, so within a tile row the K blocks are contiguous. Lane l = 16h + r of the
 *           wave reading it owns bytes [16l, 16l + 16), and byte 8j + e of those is slot e of k
 *           step 2kb + j: W[16t + r][32kb + 16j + 8(e>>2) + 4h + (e&3)]. That is the bf16 WMMA
 *           fragment's slot order (r4d_gdn_wmma.h), slots 0-3 in a lane's first dword of a step
 *           and 4-7 in its second, and r4d_layout.cpp's w8 order at 32 K a packet without w8a16's
 *           +128.
 *   pad     to a multiple of 256 bytes
 *   scales  ceil(N/32) rows of ldws = round_up(K/32, 8) bytes: byte [nb][kb] is the E8M0 exponent
 *           s of rows [32nb, 32nb + 32) x K [32kb, 32kb + 32), the value 2^(s-127), with s in
 *           [R4D_BF16F8_SMIN, R4D_BF16F8_SMAX] (127 for an all-zero block). Row padding is 0.
 *   pad     to a multiple of 256 bytes
 *
 * Element (n, k) is r4d_bf16f8_expand(code[r4d_bf16f8_slot(K, n, k)], scale[n/32][k/32]). The
 * exponent bounds make every such product a normal bf16 or a signed zero, so the kernel's f32
 * product is exact and never meets a denormal. */
#pragma once
#include <stdint.h>

#if defined(__HIPCC__) || defined(__HIP__)
#define R4D_BF16F8_HD __host__ __device__ __forceinline__
#else
#define R4D_BF16F8_HD static inline
#endif

enum { R4D_BF16F8_BLOCK = 32, R4D_BF16F8_SMIN = 10, R4D_BF16F8_SMAX = 246 };

/* Scale bytes a row of 32-row blocks: K/32, padded so every row starts on 8 bytes. */
R4D_BF16F8_HD int64_t r4d_bf16f8_ldws(int64_t K) { return (K / 32 + 7) / 8 * 8; }
/* Where the scales start: past the codes, on 256 bytes. */
R4D_BF16F8_HD int64_t r4d_bf16f8_scale_off(int64_t N, int64_t K) { return (N * K + 255) / 256 * 256; }
/* The whole stored form. */
R4D_BF16F8_HD int64_t r4d_bf16f8_bytes(int64_t N, int64_t K) {
    return r4d_bf16f8_scale_off(N, K) + (((N + 31) / 32) * r4d_bf16f8_ldws(K) + 255) / 256 * 256;
}
/* The byte holding element (n, k)'s code. */
R4D_BF16F8_HD int64_t r4d_bf16f8_slot(int64_t K, int64_t n, int64_t k) {
    const int64_t t = n / 16, r = n % 16, kb = k / 32, kin = k % 32;
    const int64_t j = kin / 16, kk = kin % 16, h = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    return ((t * (K / 32) + kb) * 32 + h * 16 + r) * 16 + j * 8 + e;
}
/* E4M3 code b times 2^(s-127) as bf16 bits: what the kernel's exact fp8 decode, its f32 product
 * with 2^(s-127) and the product's high half give. Defined for b other than the NaN codes 0x7F,
 * 0xFF and s in [R4D_BF16F8_SMIN, R4D_BF16F8_SMAX]; signed zeros stay signed. */
R4D_BF16F8_HD uint16_t r4d_bf16f8_expand(uint32_t b, uint32_t s) {
    const uint32_t sign = (b & 0x80u) << 8, mag = b & 0x7Fu;
    if (mag == 0) return (uint16_t)sign;
    if ((mag >> 3) == 0) {          /* subnormal m * 2^-9: normalise its three bits */
        const uint32_t p = mag >= 4 ? 2u : (mag >= 2 ? 1u : 0u);
        return (uint16_t)(sign | ((s - 9u + p) << 7) | (((mag << (3u - p)) & 7u) << 4));
    }
    return (uint16_t)(sign | ((mag << 4) + ((s - 7u) << 7)));
}

/* ---- the encoder (the relayout's; host code) */

/* bf16 bits x's code at exponent s, or -1 when x is no E4M3 code times 2^(s-127): a normal code
 * read off x's exponent and mantissa (its low four bits zero, an exponent 1..15, never the NaN
 * 0x7F), a zero, or one of the seven subnormals. Checked at the end: the code expands to x, which
 * is the definition of exact, so a bf16 subnormal, an infinity and a NaN are all refused. */
R4D_BF16F8_HD int r4d_bf16f8_encode(uint32_t x, uint32_t s) {
    const int a = (int)(x & 0x7FFFu), d = a - (int)((s - 7u) << 7);
    int c;
    if (d >= 0x80 && d <= 0x7E0 && (d & 0xF) == 0) {
        c = d >> 4;
    } else if (a == 0) {
        c = 0;
    } else {
        for (c = 1; c < 8; ++c)
            if (r4d_bf16f8_expand((uint32_t)c, s) == (uint32_t)a) break;
        if (c == 8) return -1;
    }
    c |= (int)((x >> 8) & 0x80u);
    return r4d_bf16f8_expand((uint32_t)c, s) == (x & 0xFFFFu) ? c : -1;
}

/* The exponent of a block of `rows` rows x 32 bf16 values at row pitch `pitch`: the smallest s with
 * every |x| <= 448 * 2^(s-127), at least R4D_BF16F8_SMIN; 127 for a block of zeros. A result past
 * R4D_BF16F8_SMAX means the block cannot be stored. x = (1 + m/128) * 2^(E-127) fits iff E - 8 < s,
 * or E - 8 == s and m <= 0x60 (448 = 1.75 * 2^8). A larger s only shifts low bits out of the codes,
 * so if this s cannot hold the block exactly, no s can. */
R4D_BF16F8_HD int r4d_bf16f8_block_exp(const uint16_t* w, int64_t pitch, int rows) {
    int need = -8;   /* below any value's: E - 8 >= -7 for every nonzero exponent field */
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < R4D_BF16F8_BLOCK; ++j) {
            const int a = w[i * pitch + j] & 0x7FFF, E = a >> 7;
            if (E == 0) continue;   /* +-0; a bf16 subnormal is refused when it is encoded */
            const int q = E - 8 + ((a & 0x7F) > 0x60 ? 1 : 0);
            if (q > need) need = q;
        }
    return need == -8 ? 127 : (need < R4D_BF16F8_SMIN ? R4D_BF16F8_SMIN : need);
}
