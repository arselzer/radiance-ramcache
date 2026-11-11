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
 * Every weight layout gives the same values: llama.cpp's block_mxfp4 rows (what the card and the
 * RAM cache hold), the release's planes (what a fresh read from the container gives) and the
 * packed form below (a CPU-only RAM cache). */
#pragma once
#include <cstddef>
#include <cstdint>

namespace ds41hx {

/* One MXFP4 matrix [N, K], K a multiple of 32. Set `blocks` (llama.cpp block_mxfp4: an e8m0 byte
 * then 16 code bytes per 32 values, value j < 16 in the low nibble of byte j and j + 16 in its
 * high nibble; K / 32 * 17 bytes a row) or `codes` and `scales` (the release's planes: [N, K / 2]
 * bytes with element 2i in the low nibble, and [N, K / 32] e8m0), or `codes` and `pk` (the packed
 * form: packed_view fills these). */
struct Mx {
    int N = 0, K = 0;
    const uint8_t* blocks = nullptr;
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
    /* Packed (below): each block's e8m0 is e0 + c for its 2-bit code c in `pk`, except in the
     * n_exc rounds listed in exc_idx (ascending n * rounds + r), whose e8m0 bytes are in exc_raw. */
    const uint8_t*  pk = nullptr;
    const uint32_t* exc_idx = nullptr;
    const uint8_t*  exc_raw = nullptr;
    int     n_exc = 0;
    uint8_t e0 = 0;
};

/* ---- the packed form: one weight in a CPU-only RAM cache slot --------------------------------
 *
 * The scales of the release's experts sit near 2^-7: nearly every block's e8m0 is 120 or 121, a
 * few 119 or 122. So a weight keeps the release's codes plane unchanged and stores each block's
 * e8m0 as a 2-bit code c relative to one base for the whole weight (e = e0 + c); the kernel's
 * unit, a round of 32 consecutive blocks of a row (a row's last round holds the rest), is stored
 * raw when any of its blocks lies outside e0 .. e0 + 3. A round of t blocks takes (t + 3) / 4
 * bytes, block j's code in bits 2 (j / tb) .. +1 of byte j % tb (tb that byte count), so a full
 * round's 8 bytes widen with one shift of each 64-bit lane. Sections, each at a 64-byte boundary
 * of the slot:
 *
 *   header   64 bytes: magic, version, N, K, cap, n_exc, e0 and the total size
 *   codes    [N, K / 2], the release's codes plane
 *   pk       [N, pk_row_bytes(K)] 2-bit codes (0 in a raw round)
 *   exc_idx  cap u32: the raw rounds' indices n * rounds + r, ascending (n_exc used)
 *   exc_raw  cap x 32 bytes: those rounds' e8m0 bytes, in block order
 *
 * The kernels widen a round's codes into its e8m0 bytes and run the planar round code unchanged,
 * so a packed weight gives the planar weight's bits. A slot has a fixed size per (N, K, cap); a
 * weight with more than cap raw rounds does not fit it. */
constexpr int kPackOk = 0, kPackOverBudget = -1, kPackInvalid = -2;
struct PackInfo { int e0 = 0; int n_exc = 0; };
size_t pk_row_bytes(int K);
/* The slot size: 0 for an unusable shape (K not a multiple of 32, more than 2^32 rounds in all,
 * cap < 0 or more than the weight's rounds). */
size_t packed_bytes(int N, int K, int cap);
/* The base for scales [N, K / 32]: the e0 in 0 .. 252 that leaves the fewest raw rounds (the lowest
 * such e0), and that count. False for an unusable shape. Writes nothing: a caller can check a
 * weight against a slot's cap before it takes the slot. */
bool pack_plan(const uint8_t* scales, int N, int K, PackInfo* plan);
/* How pack_planes writes the codes plane, a slot's bulk (11.8 of a gate_up's 12.0 MB): with `nt`
 * by non-temporal stores (copy_nt), in pieces of `slice` bytes (0: one piece), before(ctx, bytes)
 * called before each piece with its size, the first included: a background insert gives way to the
 * expert pool there. The bytes written are memcpy's. With `nt` every piece ends in an sfence, so the
 * codes are in memory, ordered before anything the caller stores after pack_planes returns (the
 * slot's publication). */
struct PackCopy {
    bool   nt = false;
    size_t slice = 0;
    void (*before)(void* ctx, size_t bytes) = nullptr;
    void*  ctx = nullptr;
};
/* Packs planes (codes [N, K / 2], scales [N, K / 32]) with pack_plan's `plan` for these scales into
 * dst (dst_bytes >= packed_bytes(N, K, cap)), every byte of the slot written; `copy` as above (null:
 * one memcpy). kPackOverBudget when more than cap rounds are raw, kPackInvalid for an unusable shape
 * or a plan that does not match the scales; dst is then unspecified. */
int  pack_planes(const uint8_t* codes, const uint8_t* scales, int N, int K, int cap,
                 const PackInfo& plan, uint8_t* dst, size_t dst_bytes, const PackCopy* copy = nullptr);
/* memcpy(dst, src, n), dst's whole 64-byte lines by non-temporal stores (neither read first nor kept
 * in the caches) and its partial ends by ordinary ones; ends in an sfence. */
void copy_nt(uint8_t* dst, const uint8_t* src, size_t n);
/* The view of the packed weight at p (`bytes` long): false unless its header describes exactly a
 * [N, K] weight of packed_bytes(N, K, cap) inside those bytes. */
bool packed_view(const uint8_t* p, size_t bytes, int N, int K, Mx* out);
/* The round trip: true when every block's e8m0, as the AVX2 kernels widen it from the packed form
 * (raw rounds included, each used once and in order), equals `scales` [N, K / 32] byte for byte. */
bool packed_scales_equal(const Mx& pk, const uint8_t* scales);
/* The inverse of the packing's scales: every block's e8m0, as the kernels widen it from the packed form
 * (raw rounds included), written to scales [N, K / 32]. With the codes plane, which the packed form keeps
 * as it is (w.codes), these are the release's planes bit for bit: a packed weight can be relaid into
 * block_mxfp4 exactly as the drive's planes are. False for an unusable view or raw rounds that are not
 * each used once and in order (scales then unspecified). */
bool packed_scales(const Mx& pk, uint8_t* scales);

uint16_t bf16_rne_nan(float f);   /* libr4d bf16_hi_rne_nan(f) >> 16 */
uint16_t f32_to_bf16(float f);    /* libr4d f32_to_bf16 */
float    bf16_to_f32(uint16_t h);

/* y[n] for n in [n0, n1): the device decode GEMV's bits. `x` is K bf16 values. */
void gemv_ref(const Mx& w, const uint16_t* x, uint16_t* y, int n0, int n1);
/* The same bits, AVX2: eight rows at once, one row a vector lane. n0 and n1 multiples of 8.
 * Every layout, the packed one included. */
void gemv_avx2(const Mx& w, const uint16_t* x, uint16_t* y, int n0, int n1);
/* gemv_avx2 for `rows` token rows of one matrix, x[r] -> y[r], each with gemv_avx2's bits: up to
 * kFewRows rows through gemv_avx2_few, more through gemv_avx2_many. */
void gemv_avx2_rows(const Mx& w, const uint16_t* const* x, uint16_t* const* y, int rows, int n0, int n1);
/* Each round of codes decoded once for up to kFewRows rows (more go in calls of kFewRows), and each
 * row's sums then run gemv_avx2's own round code. */
constexpr int kFewRows = 4;
void gemv_avx2_few(const Mx& w, const uint16_t* const* x, uint16_t* const* y, int rows, int n0, int n1);
/* gemv_avx2_few in two halves, for a caller that runs many row ranges of one matrix on the same x
 * (the pool's tiles of one job): few_lanes lays x out in the kernel's lanes, as every gemv_avx2_few
 * call does first (halved, transposed; few_lanes_floats(K) floats a row, row i at xt + i *
 * few_lanes_floats(K), xt 32-byte aligned), and gemv_avx2_few_lanes runs [n0, n1) on them, with
 * gemv_avx2_few's bits. rows 1 .. kFewRows. */
size_t few_lanes_floats(int K);
void few_lanes(const uint16_t* const* x, int rows, int K, float* xt);
void gemv_avx2_few_lanes(const Mx& w, const float* xt, uint16_t* const* y, int rows, int n0, int n1);
/* Up to kGemvRows rows at once (more go in calls of kGemvRows), for a prefill pass's many rows of one
 * expert: the output rows in blocks of 16, each round of each of them decoded once into f32 weights,
 * the token rows' sums in register blocks of two output rows by four token rows. */
constexpr int kGemvRows = 32;
void gemv_avx2_many(const Mx& w, const uint16_t* const* x, uint16_t* const* y, int rows, int n0, int n1);

/* swiglu_clamped over one row: gu is [gate n | up n], y is n. `silu` has 65536 entries. */
void swiglu(const float* silu, const uint16_t* gu, uint16_t* y, int n, float limit);
/* The same, eight values a step (the pool's form: one job's swiglu sits between its gate_up and its
 * down tiles). The same multiply and rounding; the clamp is maxps/minps, which return the limit
 * where fmax/fmin do for a quiet NaN `up`. Identical except for a signalling-NaN `up`, which the
 * gate_up GEMV never writes (its NaNs are quiet). */
void swiglu_avx2(const float* silu, const uint16_t* gu, uint16_t* y, int n, float limit);

/* The table from the host's expf. For tests of everything else only: the served table must be
 * captured from the device (ds41_mxfp4_device_check.hip). */
void silu_table_host(float* table, float limit);

/* One expert for one token: gate_up [2 * n_ff], act [n_ff], down [n_embd]. Rows are split over
 * `threads` workers with a join between the projections (threads <= 1 runs inline). */
struct Expert { Mx gu, dn; };
void expert_forward(const Expert& e, const float* silu, float limit, const uint16_t* x,
                    uint16_t* gate_up, uint16_t* act, uint16_t* down, int threads, bool avx2);

}  // namespace ds41hx
