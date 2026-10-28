/* ds41_host_experts.h -- DeepSeek-V4.1 routed experts computed on the host, bit-identical to the
 * device decode path, so an expert held in host memory need not be copied to the card.
 *
 * The device chain for one decode token and one expert (arch/common/rad_block_moe_fp8.h,
 * act_bf16 path) is three ops, each rounded to bf16:
 *
 *   gate_up = moe_gemm_ggml(x, W_gu)           libr4d r4d_ds41_mxfp4.hip, gemv path
 *   act     = swiglu_clamped(gate_up, limit)    libr4d r4d_glm.hip
 *   down    = moe_gemm_ggml(act, W_dn)          libr4d r4d_ds41_mxfp4.hip, gemv path
 *
 * and moe_gather applies the routing weights afterwards, unchanged. Here each op repeats the
 * device's sequence of f32 operations:
 *
 *   gemv     w = kv[nibble] * 2^(e-128), exact; for output row n, lane L of 32 sums the blocks
 *            L, L+32, L+64, ... in increasing order, each block's 32 values in mx_unit's group
 *            order (low nibbles of bytes 0..7, of 8..15, then the high nibbles of each), one
 *            rounded product and one rounded sum a value (no FMA); then r4d_wave_sum's butterfly
 *            (xor 16, 8, 4, 2, 1) and bf16_hi_rne_nan.
 *   swiglu   g = fminf(gate, limit), u = fminf(fmaxf(up, -limit), limit),
 *            f32_to_bf16(silu(g) * u). silu(g) = g / (1 + expf(-g)) uses the device's expf, which
 *            the host cannot reproduce, so it comes from a table of the device's own results,
 *            one entry per bf16 gate pattern, captured once on the card.
 *
 * Both weight layouts give the same values: llama.cpp's block_mxfp4 rows (what the card and the
 * RAM cache hold) and the release's planes (what a fresh read from the container gives). */
#pragma once
#include <cstddef>
#include <cstdint>

namespace ds41hx {

/* One MXFP4 matrix [N, K], K a multiple of 32. Set `blocks` (llama.cpp block_mxfp4: an e8m0 byte
 * then 16 code bytes per 32 values, value j < 16 in the low nibble of byte j and j + 16 in its
 * high nibble; K / 32 * 17 bytes a row) or `codes` and `scales` (the release's planes: [N, K / 2]
 * bytes with element 2i in the low nibble, and [N, K / 32] e8m0). */
struct Mx {
    int N = 0, K = 0;
    const uint8_t* blocks = nullptr;
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
};

uint16_t bf16_rne_nan(float f);   /* libr4d bf16_hi_rne_nan(f) >> 16 */
uint16_t f32_to_bf16(float f);    /* libr4d f32_to_bf16 */
float    bf16_to_f32(uint16_t h);

/* y[n] for n in [n0, n1): the device decode GEMV's bits. `x` is K bf16 values. */
void gemv_ref(const Mx& w, const uint16_t* x, uint16_t* y, int n0, int n1);
/* The same bits, AVX2: eight rows at once, one row a vector lane. n0 and n1 multiples of 8. */
void gemv_avx2(const Mx& w, const uint16_t* x, uint16_t* y, int n0, int n1);

/* swiglu_clamped over one row: gu is [gate n | up n], y is n. `silu` has 65536 entries. */
void swiglu(const float* silu, const uint16_t* gu, uint16_t* y, int n, float limit);

/* The table from the host's expf. For tests of everything else only: the served table must be
 * captured from the device (ds41_mxfp4_device_check.hip). */
void silu_table_host(float* table, float limit);

/* One expert for one token: gate_up [2 * n_ff], act [n_ff], down [n_embd]. Rows are split over
 * `threads` workers with a join between the projections (threads <= 1 runs inline). */
struct Expert { Mx gu, dn; };
void expert_forward(const Expert& e, const float* silu, float limit, const uint16_t* x,
                    uint16_t* gate_up, uint16_t* act, uint16_t* down, int threads, bool avx2);

}  // namespace ds41hx
