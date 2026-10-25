// R4D -- DeepSeek-V4.1's elem ops. Each kernel rounds where its libref function does.
//
//   token_map            mapped[m] = table[ids[m]]                         (ref_sparse_kv.cpp)
//   stream_gate_add      Engram's normalised-dot gate into hc streams      (ref_sparse_kv.cpp)
//   rope_rows            rotary from a rope_table plane over a head window (ref_attn.cpp)
//   quant_roundtrip      block-scaled fp8 / fp4 quantise-dequantise        (ref_sparse_kv.cpp)
//   mhc_mix              mHC projection, gates and Sinkhorn, pre out       (ref_mhc.cpp)
//   mhc_collapse         x = sum_j pre[j] h[j]                             (ref_mhc.cpp)
//   softmax_pool         channel-wise softmax pooling of ratio rows        (ref_sparse_kv.cpp)
//   entry_store          one entry a block into a paged cache              (ref_sparse_kv.cpp)
//   router_scored_topk   correction-bias routing over named scores         (ref_glm.cpp)
//   embed_lookup_planes  gather from affine code/scale(/table) planes      (ref_gemm.cpp)
//
// Every product and sum is the one libref forms, unfused (-ffp-contract=off), in f32; only the
// order of a long sum differs (a lane's strided partial, then a fixed butterfly). Stores narrow as
// rad_f32_to_bf16 does (RNE, a NaN quieted). libref's transcendentals are glibc's float expf /
// log1pf or ref_exp (exp in double, rounded once); here each is the double routine rounded once,
// which is the correctly rounded value glibc returns in all but near-tie cases.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_plugin.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

/* ==================================================================== libref's scalar rules */

// rad_f32_to_bf16 (abi/rad_plugin.h): round to nearest even, a NaN quieted rather than rounded.
__device__ __forceinline__ uint16_t ds_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float ds_bf(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// rad_f16_to_f32, bit for bit (subnormals renormalised, a NaN's payload kept).
__device__ __forceinline__ float ds_f16(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    return __uint_as_float(u);
}

// rad_fp8e4m3_to_f32.
__device__ __forceinline__ float ds_e4m3(uint32_t b) {
    const uint32_t s = (b >> 7) << 31;
    const uint32_t e = (b >> 3) & 0xfu;
    uint32_t m = b & 0x7u, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

// rad_f32_to_fp8e4m3: RNE, saturating at 448, NaN to 0x7f.
__device__ __forceinline__ uint32_t ds_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t s = (u >> 24) & 0x80u;
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7fu;
    if (a >= 0x43e00000u) return s | 0x7eu;
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        uint32_t q = m >> sh;
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return s | q;
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee = e + 7;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return s | 0x7eu;
    return s | ((uint32_t)ee << 3) | q;
}

// ref e8m0_to_f32: 2^(e-127), 0xff NaN (e 0 is the subnormal 2^-127).
__device__ __forceinline__ float ds_e8m0(uint32_t e) {
    if (e == 0xffu) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(e ? e << 23 : 0x00400000u);
}

// std::max(a, b) and std::clamp as libstdc++ evaluates them (a NaN `b` is dropped by max, kept by
// clamp), which fmaxf/fminf would not reproduce.
__device__ __forceinline__ float ds_max(float a, float b) { return (a < b) ? b : a; }
__device__ __forceinline__ float ds_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (hi < v ? hi : v);
}

__device__ __forceinline__ float ds_exp(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float ds_log1p(float x) { return (float)log1p((double)x); }

// An element of a float plane of any of the dtypes ld_dt reads that these ops accept.
__device__ __forceinline__ float ds_ld(uint32_t dt, const void* p, long i) {
    switch (dt) {
        case RAD_F32:    return ((const float*)p)[i];
        case RAD_F16:    return ds_f16(((const uint16_t*)p)[i]);
        case RAD_BF16:   return ds_bf(((const uint16_t*)p)[i]);
        case RAD_F8E4M3: return ds_e4m3(((const uint8_t*)p)[i]);
        case RAD_E8M0:   return ds_e8m0(((const uint8_t*)p)[i]);
        case RAD_I8:     return (float)((const int8_t*)p)[i];
        case RAD_U8:     return (float)((const uint8_t*)p)[i];
        default:         return 0.0f;
    }
}

/* ==================================================================== host-side operand checks */

// An index operand read as libref's offlin(t, m): element m at m * step. Contiguous, or one
// non-unit axis with its stride.
bool lin_step(const RadTensor& t, long* step) {
    if (!t.data || t.rank == 0) return false;
    if (rad_tensor_is_contiguous(&t)) { *step = 1; return true; }
    int nd = 0, d = 0;
    for (uint32_t i = 0; i < t.rank; ++i)
        if (t.shape[i] != 1) { ++nd; d = (int)i; }
    if (nd != 1) return false;
    *step = (long)t.stride[d];
    return true;
}
// [rows, cols] with a unit inner stride; the row pitch is stride[0].
bool mat(const RadTensor& t, uint32_t dt, long rows, long cols) {
    return t.data && t.dtype == dt && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols &&
           t.stride[1] == 1 && t.stride[0] >= cols;
}
// [rows, cols] of any inner stride (the few f32 coefficients a row of mhc_mix writes).
bool mat_s(const RadTensor& t, uint32_t dt, long rows, long cols) {
    return t.data && t.dtype == dt && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}
long last_stride(const RadTensor& t) { return t.rank ? (long)t.stride[t.rank - 1] : 1; }
unsigned grid_rows(long rows) { return (unsigned)(rows < 65535 ? rows : 65535); }

/* ==================================================================== token_map */

__global__ __launch_bounds__(256) void token_map_kernel(const int32_t* ids, long is,
                                                        const int32_t* tab, long ts, int32_t* y,
                                                        long ys, long M, long nv) {
    for (long m = (long)blockIdx.x * 256 + threadIdx.x; m < M; m += (long)gridDim.x * 256) {
        const long id = ids[m * is];
        y[m * ys] = id < 0 || id >= nv ? -1 : tab[id * ts];
    }
}

int token_map_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &tab = a->t[1], &y = a->t[2];
    const long nv = (long)r4d_geti_or(a, "n_vocab", 0);
    const long M = (long)rad_tensor_numel(&x);
    if (nv < 1 || rad_tensor_numel(&tab) != nv || M < 1 || rad_tensor_numel(&y) != M)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_I32 || tab.dtype != RAD_I32 || y.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    long is, ts, ys;
    if (!lin_step(x, &is) || !lin_step(tab, &ts) || !lin_step(y, &ys)) return R4D_NO(RAD_E_SHAPE);
    const long blocks = (M + 255) / 256;
    return R4D_EMIT(stream, token_map_kernel, dim3((unsigned)(blocks < 65535 ? blocks : 65535)),
                    dim3(256), 0, (const int32_t*)x.data, is, (const int32_t*)tab.data, ts,
                    (int32_t*)y.data, ys, M, nv);
}

/* ==================================================================== stream_gate_add
 * One workgroup a (row, stream): the three sums over the stream's n values (a lane's strided
 * partial, the wave butterfly, the eight waves in order), the gate on one lane, then the write. */

__global__ __launch_bounds__(256) void stream_gate_kernel(const uint16_t* h, long hp,
                                                          const uint16_t* k, long kp,
                                                          const uint16_t* v, long vp,
                                                          const float* w, long wst, uint16_t* out,
                                                          long op, int n, long rows, float eps,
                                                          float ge) {
    __shared__ float red[3][8];
    __shared__ float gate_s;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, s = blockIdx.x;
    const float* ws = w + (long)s * n * wst;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        const uint16_t* hr = h + r * hp + (long)s * n;
        const uint16_t* kr = k + r * kp + (long)s * n;
        float sh = 0, sk = 0, dot = 0;
        for (int i = tid; i < n; i += 256) {
            const float hv = ds_bf(hr[i]), kv = ds_bf(kr[i]);
            sh += hv * hv;
            sk += kv * kv;
            dot += hv * ws[(long)i * wst] * kv;
        }
        sh = r4d_wave_sum(sh);
        sk = r4d_wave_sum(sk);
        dot = r4d_wave_sum(dot);
        if (!lane) {
            red[0][wave] = sh;
            red[1][wave] = sk;
            red[2][wave] = dot;
        }
        __syncthreads();
        if (!tid) {
            float a = 0, b = 0, c = 0;
            for (int q = 0; q < 8; ++q) {
                a += red[0][q];
                b += red[1][q];
                c += red[2][q];
            }
            const float rstd =
                (1.0f / sqrtf(a / (float)n + eps)) * (1.0f / sqrtf(b / (float)n + eps));
            c = c * rstd * (1.0f / sqrtf((float)n));
            const float t = copysignf(sqrtf(ds_max(fabsf(c), ge)), c);
            gate_s = 1.0f / (1.0f + ds_exp(-t));
        }
        __syncthreads();
        const float g = gate_s;
        const uint16_t* vr = v + r * vp;
        uint16_t* orow = out + r * op + (long)s * n;
        for (int i = tid; i < n; i += 256) orow[i] = ds_bf16(ds_bf(hr[i]) + g * ds_bf(vr[i]));
        __syncthreads();
    }
}

int stream_gate_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor &h = a->t[0], &k = a->t[1], &v = a->t[2], &w = a->t[3], &out = a->t[4];
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    const float eps = (float)r4d_getf(a, "eps", -1.0), ge = (float)r4d_getf(a, "gate_eps", -1.0);
    if (n < 1 || hc < 1 || hc > 16 || n * hc > INT_MAX || !(eps >= 0) || !(ge >= 0) ||
        h.rank != 2 || h.shape[0] < 1)
        return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || k.dtype != RAD_BF16 || v.dtype != RAD_BF16 || w.dtype != RAD_F32 ||
        out.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat(k, RAD_BF16, rows, hc * n) ||
        !mat(v, RAD_BF16, rows, n) || !mat(out, RAD_BF16, rows, hc * n) || !w.data ||
        rad_tensor_numel(&w) != hc * n || !(w.rank == 1 || rad_tensor_is_contiguous(&w)))
        return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, stream_gate_kernel, dim3((unsigned)hc, grid_rows(rows)), dim3(256), 0,
                    (const uint16_t*)h.data, (long)h.stride[0], (const uint16_t*)k.data,
                    (long)k.stride[0], (const uint16_t*)v.data, (long)v.stride[0],
                    (const float*)w.data, last_stride(w), (uint16_t*)out.data, (long)out.stride[0],
                    (int)n, rows, eps, ge);
}

/* ==================================================================== rope_rows
 * One thread a rotated pair of a row, rows over grid.y. */

__global__ __launch_bounds__(256) void rope_rows_kernel(uint16_t* x, long xp, const float* cs,
                                                        long cp, long cst, const int32_t* pos,
                                                        long ps, long M, long tab_rows, int hd,
                                                        int half, int off, int pairs, int gptj,
                                                        float sign) {
    const int q = blockIdx.x * 256 + threadIdx.x;
    if (q >= pairs) return;
    const int hh = q / half, i = q - hh * half;
    const long ia = (long)hh * hd + off + (gptj ? 2 * i : i), ib = gptj ? ia + 1 : ia + half;
    for (long m = blockIdx.y; m < M; m += gridDim.y) {
        const long p = pos[m * ps];
        if (p < 0 || p >= tab_rows) continue;
        const float c = cs[p * cp + (long)i * cst], s = sign * cs[p * cp + (long)(half + i) * cst];
        uint16_t* row = x + m * xp;
        const float va = ds_bf(row[ia]), vb = ds_bf(row[ib]);
        row[ia] = ds_bf16(va * c - vb * s);
        row[ib] = ds_bf16(va * s + vb * c);
    }
}

int rope_rows_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &cs = a->t[1], &pos = a->t[2];
    const long nh = (long)r4d_geti_or(a, "n_head", 0), hd = (long)r4d_geti_or(a, "head_dim", 0);
    const long rot = (long)r4d_geti_or(a, "rot", 0), off = (long)r4d_geti_or(a, "off", -1);
    const long inv = (long)r4d_geti_or(a, "inverse", -1);
    const char* mode = r4d_gets(a, "mode");
    if (!mode) return R4D_NO(RAD_E_UNSUPPORTED);
    const bool gptj = !std::strcmp(mode, "gptj");
    if (!gptj && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (nh < 1 || hd < 2 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd ||
        (inv != 0 && inv != 1) || nh * hd > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || cs.dtype != RAD_F32 || pos.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (x.rank != 2 || x.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long M = x.shape[0];
    long ps;
    if (!mat(x, RAD_BF16, M, nh * hd) || cs.rank != 2 || cs.shape[1] != rot ||
        rad_tensor_numel(&pos) != M || !lin_step(pos, &ps))
        return R4D_NO(RAD_E_SHAPE);
    const long half = rot / 2, pairs = nh * half;
    return R4D_EMIT(stream, rope_rows_kernel, dim3((unsigned)((pairs + 255) / 256), grid_rows(M)),
                    dim3(256), 0, (uint16_t*)x.data, (long)x.stride[0], (const float*)cs.data,
                    (long)cs.stride[0], (long)cs.stride[1], (const int32_t*)pos.data, ps, M,
                    (long)cs.shape[0], (int)hd, (int)half, (int)off, (int)pairs, gptj ? 1 : 0,
                    inv ? -1.0f : 1.0f);
}

/* ==================================================================== quant_roundtrip
 * A block's amax, its scale and the round trip of each value, as ref_quant_roundtrip's
 * block_scale and roundtrip. A power-of-two block up to 32 is a lane group (one value a thread, the
 * max over the group's lanes); any other block is one thread a block. amax is a max, so both give
 * the same bits. */

enum { kFp8 = 0, kFp4 = 1 };
enum { kUe8m0 = 0, kE4m3 = 1, kScaleF32 = 2 };

__device__ __forceinline__ int ds_ceil_log2(float v) {
    const uint32_t u = __float_as_uint(v);
    return (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
}
__device__ __forceinline__ float ds_pow2(int e) { return __uint_as_float((uint32_t)(e + 127) << 23); }

__device__ __forceinline__ float block_scale(float amax, int fmt, int sc) {
    if (fmt == kFp8) {
        amax = ds_max(amax, 1e-4f);
        const float inv = 1.0f / 448.0f;
        return sc == kUe8m0 ? ds_pow2(ds_ceil_log2(amax * inv)) : amax * inv;
    }
    if (sc == kE4m3) {
        amax = ds_max(amax, 6.0f * 0x1p-9f);
        return ds_e4m3(ds_to_e4m3(amax / 6.0f));
    }
    amax = ds_max(amax, 6.0f * 0x1p-126f);
    const float inv = 1.0f / 6.0f;
    return sc == kUe8m0 ? ds_pow2(ds_ceil_log2(amax * inv)) : amax * inv;
}

// ref e2m1_code then mxfp4_elem_to_f32: the nearest of the eight magnitudes, ties to the even
// code, the sign kept (so -0 for a small negative), NaN to +0.
__device__ __forceinline__ float ds_e2m1_rt(float v) {
    if (v != v) return 0.0f;
    const bool neg = (__float_as_uint(v) >> 31) != 0;
    const float a = fabsf(v);
    const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float bm = 0.0f;
#pragma unroll
    for (int c = 1; c < 8; ++c) {
        const float d = fabsf(mag[c] - a), db = fabsf(bm - a);
        if (d < db || (d == db && (c & 1) == 0)) bm = mag[c];
    }
    return neg ? -bm : bm;
}

__device__ __forceinline__ float roundtrip(float v, float scale, int fmt) {
    if (fmt == kFp8) return ds_e4m3(ds_to_e4m3(ds_clamp(v / scale, -448.0f, 448.0f))) * scale;
    return ds_e2m1_rt(ds_clamp(v / scale, -6.0f, 6.0f)) * scale;
}

template <int B>
__global__ __launch_bounds__(256) void quant_group_kernel(uint16_t* x, long xp, int n, long rows,
                                                          int fmt, int sc) {
    const int col = blockIdx.x * 256 + threadIdx.x;
    const bool in = col < n;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* row = x + r * xp;
        const float v = in ? ds_bf(row[col]) : 0.0f;
        float a = fabsf(v);
        a = a == a ? a : 0.0f;  // std::max(amax, NaN) keeps amax: a NaN never wins
        const float amax = r4d_xor_max<B>(a);
        if (in) row[col] = ds_bf16(roundtrip(v, block_scale(amax, fmt, sc), fmt));
    }
}

__global__ __launch_bounds__(256) void quant_serial_kernel(uint16_t* x, long xp, int n, int block,
                                                           long rows, int fmt, int sc) {
    const int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= n / block) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* p = x + r * xp + (long)b * block;
        float amax = 0;
        for (int i = 0; i < block; ++i) amax = ds_max(amax, fabsf(ds_bf(p[i])));
        const float s = block_scale(amax, fmt, sc);
        for (int i = 0; i < block; ++i) p[i] = ds_bf16(roundtrip(ds_bf(p[i]), s, fmt));
    }
}

int quant_rt_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 1 || !a->t[0].data) return RAD_E_INVAL;
    const RadTensor& x = a->t[0];
    const long n = (long)r4d_geti_or(a, "n", 0), block = (long)r4d_geti_or(a, "block", 0);
    const char* fs = r4d_gets(a, "format");
    const char* ss = r4d_gets(a, "scale");
    if (!fs || !ss) return R4D_NO(RAD_E_UNSUPPORTED);
    int fmt, sc;
    if (!std::strcmp(fs, "fp8_e4m3")) fmt = kFp8;
    else if (!std::strcmp(fs, "fp4_e2m1")) fmt = kFp4;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (!std::strcmp(ss, "ue8m0")) sc = kUe8m0;
    else if (!std::strcmp(ss, "e4m3") && fmt == kFp4) sc = kE4m3;
    else if (!std::strcmp(ss, "f32") && fmt == kFp8) sc = kScaleF32;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (n < 1 || block < 1 || n % block || n > INT_MAX || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != n)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long rows = x.shape[0];
    if (!mat(x, RAD_BF16, rows, n)) return R4D_NO(RAD_E_SHAPE);
    uint16_t* xd = (uint16_t*)x.data;
    const long xp = (long)x.stride[0];
    const dim3 grid((unsigned)((n + 255) / 256), grid_rows(rows));
#define QRT_GROUP(B)                                                                               \
    case B:                                                                                        \
        return R4D_EMIT(stream, quant_group_kernel<B>, grid, dim3(256), 0, xd, xp, (int)n, rows,   \
                        fmt, sc)
    switch (block) {
        QRT_GROUP(1);
        QRT_GROUP(2);
        QRT_GROUP(4);
        QRT_GROUP(8);
        QRT_GROUP(16);
        QRT_GROUP(32);
        default: break;
    }
#undef QRT_GROUP
    const long nb = n / block;
    return R4D_EMIT(stream, quant_serial_kernel, dim3((unsigned)((nb + 255) / 256), grid_rows(rows)),
                    dim3(256), 0, xd, xp, (int)n, (int)block, rows, fmt, sc);
}

/* ==================================================================== mhc_mix
 * The projection is mhc_pre's: hc*(hc+2) dots of the row with fn and its sum of squares. The row
 * splits into 1024-column chunks; a wave takes a chunk of R rows (lane L the columns L, L+32, ..
 * in order), closes each sum with the wave butterfly, and the chunk partials are added in chunk
 * order. With scratch the chunks are separate workgroups (partials, then a finishing wave a row);
 * without, one wave a row walks the chunks itself, in the same order, for the same bits. The
 * finish is ref finish_row lane-parallel: element (j,k) of the Sinkhorn matrix on lane j*hc+k (and
 * +32 from hc 6), every row and column sum gathered by shuffles in the scalar loop's order. */

constexpr int kMixCols = 1024;

struct MixOut {
    const float* scale;
    long sst;
    const float* base;
    long bst;
    float* pre;
    long prp, prs;
    float* post;
    long pop, pos;
    float* comb;
    long cp, cs;
    int width, iters;
    float eps, heps;
};

template <int HC, int R>
__device__ __forceinline__ void mix_chunk(const uint16_t* h, long hp, const float* fn, long fp,
                                          int width, long r0, int nr, int chunk, int lane,
                                          float (&acc)[HC * (HC + 2) + 1][R]) {
    constexpr int MIX = HC * (HC + 2);
#pragma unroll
    for (int j = 0; j <= MIX; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[j][r] = 0.0f;
    const int c0 = chunk * kMixCols;
    for (int t = 0; t < kMixCols / 32; ++t) {
        const int col = c0 + t * 32 + lane;
        const bool in = col < width;
        float v[R];
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = in && r < nr ? ds_bf(h[(r0 + r) * hp + col]) : 0.0f;
#pragma unroll
        for (int j = 0; j < MIX; ++j) {
            const float w = in ? fn[(long)j * fp + col] : 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) acc[j][r] += v[r] * w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[MIX][r] += v[r] * v[r];
    }
}

// d[q] holds sum j = lane + 32q (the hc*(hc+2) dots, then the sum of squares) in every lane that
// owns one. Every lane must enter: the gathers are shuffles.
template <int HC>
__device__ __forceinline__ void mix_finish(const float (&d)[(HC * (HC + 2) + 32) / 32], int lane,
                                           long row, const MixOut& o) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32, NN = HC * HC, NE = (NN + 31) / 32;
    const float sq = __shfl(d[MIX / 32], MIX % 32, 32);
    const float inv = 1.0f / sqrtf(sq / (float)o.width + o.eps);
    float cf[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) cf[q] = d[q] * inv;
    auto coeff = [&](int j) {
        float r = 0.0f;
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float t = __shfl(cf[q], j & 31, 32);
            if ((j >> 5) == q) r = t;
        }
        return r;
    };
    const float s0 = o.scale[0], s1 = o.scale[o.sst], s2 = o.scale[2 * o.sst];
    const int jl = lane < HC ? lane : 0;
    const float cpre = coeff(jl), cpost = coeff(HC + jl);
    if (lane < HC) {
        const float pj = 1.0f / (1.0f + ds_exp(-(cpre * s0 + o.base[(long)lane * o.bst]))) + o.heps;
        const float z = cpost * s1 + o.base[(long)(HC + lane) * o.bst];
        o.post[row * o.pop + lane * o.pos] = 2.0f / (1.0f + ds_exp(-z));
        o.pre[row * o.prp + lane * o.prs] = pj;
    }
    float c[NE];
    int rb[NE], ck[NE];
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u, ee = e < NN ? e : 0;
        rb[u] = ee / HC * HC;
        ck[u] = ee % HC;
        const float cq = coeff(2 * HC + ee);
        c[u] = e < NN ? cq * s2 + o.base[(long)(2 * HC + ee) * o.bst] : 0.0f;
    }
    auto get = [&](int idx) {
        float r = 0.0f;
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            const float t = __shfl(c[u], idx & 31, 32);
            if ((idx >> 5) == u) r = t;
        }
        return r;
    };
    {
        float mx[NE], sum[NE];
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            mx[u] = -INFINITY;
#pragma unroll
            for (int k = 0; k < HC; ++k) mx[u] = ds_max(mx[u], get(rb[u] + k));
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] = ds_exp(c[u] - mx[u]);
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            sum[u] = 0.0f;
#pragma unroll
            for (int k = 0; k < HC; ++k) sum[u] += get(rb[u] + k);
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] = c[u] / sum[u] + o.heps;
    }
    for (int it = 0; it < o.iters; ++it) {
        float sum[NE];
        if (it) {
#pragma unroll
            for (int u = 0; u < NE; ++u) {
                sum[u] = 0.0f;
#pragma unroll
                for (int k = 0; k < HC; ++k) sum[u] += get(rb[u] + k);
            }
#pragma unroll
            for (int u = 0; u < NE; ++u) c[u] /= sum[u] + o.heps;
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            sum[u] = 0.0f;
#pragma unroll
            for (int j = 0; j < HC; ++j) sum[u] += get(j * HC + ck[u]);
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] /= sum[u] + o.heps;
    }
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u;
        if (e < NN) o.comb[row * o.cp + (long)e * o.cs] = c[u];
    }
}

// Partials of chunk blockIdx.x for 8*R rows a workgroup: part[row][chunk][j], j <= mix.
template <int HC, int R>
__global__ __launch_bounds__(256) void mix_partial_kernel(const uint16_t* h, long hp,
                                                          const float* fn, long fp, float* part,
                                                          int width, int chunks, long rows) {
    constexpr int MIX = HC * (HC + 2);
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, chunk = blockIdx.x;
    for (long g = blockIdx.y; g * 8 * R < rows; g += gridDim.y) {
        const long r0 = (g * 8 + wave) * R;
        if (r0 >= rows) continue;
        const int nr = (int)(rows - r0 < R ? rows - r0 : R);
        float acc[MIX + 1][R];
        mix_chunk<HC, R>(h, hp, fn, fp, width, r0, nr, chunk, lane, acc);
#pragma unroll
        for (int j = 0; j <= MIX; ++j)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float s = r4d_wave_sum(acc[j][r]);
                if (!lane && r < nr) part[((r0 + r) * chunks + chunk) * (MIX + 1) + j] = s;
            }
    }
}

template <int HC>
__global__ __launch_bounds__(128) void mix_finish_kernel(const float* part, int chunks, MixOut o,
                                                         long rows) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32;
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float d[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const int j = lane + 32 * q;
        float s = 0.0f;
        if (j <= MIX)
            for (int c = 0; c < chunks; ++c) s += part[(row * chunks + c) * (MIX + 1) + j];
        d[q] = s;
    }
    mix_finish<HC>(d, lane, row, o);
}

// Without scratch: one wave a row, the chunks in order, the same partials and the same sums.
template <int HC>
__global__ __launch_bounds__(128) void mix_fused_kernel(const uint16_t* h, long hp, const float* fn,
                                                        long fp, int chunks, MixOut o, long rows) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32;
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    float tot[MIX + 1];
#pragma unroll
    for (int j = 0; j <= MIX; ++j) tot[j] = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        float acc[MIX + 1][1];
        mix_chunk<HC, 1>(h, hp, fn, fp, o.width, row, 1, c, lane, acc);
#pragma unroll
        for (int j = 0; j <= MIX; ++j) tot[j] += r4d_wave_sum(acc[j][0]);
    }
    float d[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        d[q] = 0.0f;
#pragma unroll
        for (int j = 0; j <= MIX; ++j)
            if (j == lane + 32 * q) d[q] = tot[j];
    }
    mix_finish<HC>(d, lane, row, o);
}

long mix_chunks(long width) { return (width + kMixCols - 1) / kMixCols; }

template <int HC>
int mix_emit(RadStream stream, const RadTensor& h, const RadTensor& fn, const MixOut& o, long rows,
             void* scratch, int64_t scratch_bytes) {
    constexpr int MIX = HC * (HC + 2), R = HC <= 4 ? 2 : 1;
    const long chunks = mix_chunks(o.width);
    const int64_t need = (int64_t)rows * chunks * (MIX + 1) * (int64_t)sizeof(float);
    const unsigned fin = (unsigned)((rows + 3) / 4);
    if (scratch && scratch_bytes >= need) {
        float* part = (float*)scratch;
        const long groups = (rows + 8 * R - 1) / (8 * R);
        const int rc = R4D_EMIT(stream, (mix_partial_kernel<HC, R>),
                                dim3((unsigned)chunks, grid_rows(groups)), dim3(256), 0,
                                (const uint16_t*)h.data, (long)h.stride[0], (const float*)fn.data,
                                (long)fn.stride[0], part, o.width, (int)chunks, rows);
        if (rc < 0) return rc;
        return R4D_EMIT(stream, mix_finish_kernel<HC>, dim3(fin), dim3(128), 0, (const float*)part,
                        (int)chunks, o, rows);
    }
    return R4D_EMIT(stream, mix_fused_kernel<HC>, dim3(fin), dim3(128), 0, (const uint16_t*)h.data,
                    (long)h.stride[0], (const float*)fn.data, (long)fn.stride[0], (int)chunks, o,
                    rows);
}

bool mix_params(const RadArgs* a, long* n, long* hc, long* iters, float* eps, float* heps) {
    *n = (long)r4d_geti_or(a, "n", 0);
    *hc = (long)r4d_geti_or(a, "hc", 0);
    *iters = (long)r4d_geti_or(a, "iterations", 0);
    *eps = (float)r4d_getf(a, "eps", 0);
    *heps = (float)r4d_getf(a, "hc_eps", 0);
    return *n > 0 && *hc >= 2 && *hc <= 8 && *iters >= 1 && *iters <= 256 &&
           *hc * *n <= (1L << 30) && std::isfinite(*eps) && std::isfinite(*heps) && *eps > 0 &&
           *heps >= 0;
}

int mhc_mix_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    long n, hc, iters;
    float eps, heps;
    if (!mix_params(a, &n, &hc, &iters, &eps, &heps)) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &fn = a->t[1], &sc = a->t[2], &bs = a->t[3];
    const RadTensor &pre = a->t[4], &post = a->t[5], &comb = a->t[6];
    const long width = hc * n, mix = hc * (hc + 2);
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || fn.dtype != RAD_F32 || sc.dtype != RAD_F32 || bs.dtype != RAD_F32 ||
        pre.dtype != RAD_F32 || post.dtype != RAD_F32 || comb.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, width) || !mat(fn, RAD_F32, mix, width) || !sc.data ||
        sc.rank != 1 || sc.shape[0] != 3 || !bs.data || bs.rank != 1 || bs.shape[0] != mix ||
        !mat_s(pre, RAD_F32, rows, hc) || !mat_s(post, RAD_F32, rows, hc) ||
        !mat_s(comb, RAD_F32, rows, hc * hc))
        return R4D_NO(RAD_E_SHAPE);
    MixOut o;
    o.scale = (const float*)sc.data;
    o.sst = (long)sc.stride[0];
    o.base = (const float*)bs.data;
    o.bst = (long)bs.stride[0];
    o.pre = (float*)pre.data;
    o.prp = (long)pre.stride[0];
    o.prs = (long)pre.stride[1];
    o.post = (float*)post.data;
    o.pop = (long)post.stride[0];
    o.pos = (long)post.stride[1];
    o.comb = (float*)comb.data;
    o.cp = (long)comb.stride[0];
    o.cs = (long)comb.stride[1];
    o.width = (int)width;
    o.iters = (int)iters;
    o.eps = eps;
    o.heps = heps;
    switch (hc) {
        case 2: return mix_emit<2>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 3: return mix_emit<3>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 4: return mix_emit<4>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 5: return mix_emit<5>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 6: return mix_emit<6>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 7: return mix_emit<7>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 8: return mix_emit<8>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        default: return R4D_NO(RAD_E_SHAPE);
    }
}

/* ==================================================================== mhc_collapse
 * One column a thread (eight with 16-byte rows), the streams summed from zero in order. */

__global__ __launch_bounds__(256) void collapse_kernel(const uint16_t* h, long hp, const float* pre,
                                                       long pp, long ps, uint16_t* x, long xp,
                                                       int n, int hc, long rows) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        float v = 0;
        for (int j = 0; j < hc; ++j) v += pre[r * pp + j * ps] * ds_bf(h[r * hp + (long)j * n + i]);
        x[r * xp + i] = ds_bf16(v);
    }
}

__global__ __launch_bounds__(256) void collapse_vec_kernel(const uint16_t* h, long hp,
                                                           const float* pre, long pp, long ps,
                                                           uint16_t* x, long xp, int n, int hc,
                                                           long rows) {
    const int i = (blockIdx.x * 256 + threadIdx.x) * 8;
    if (i >= n) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < hc; ++j) {
            const float pj = pre[r * pp + j * ps];
            const uint4 q = *reinterpret_cast<const uint4*>(h + r * hp + (long)j * n + i);
            const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += pj * ds_bf((uint16_t)(w[e >> 1] >> (16 * (e & 1))));
        }
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            o[e] = (uint32_t)ds_bf16(v[2 * e]) | ((uint32_t)ds_bf16(v[2 * e + 1]) << 16);
        *reinterpret_cast<uint4*>(x + r * xp + i) = uint4{ o[0], o[1], o[2], o[3] };
    }
}

int mhc_collapse_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8 || hc * n > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &pre = a->t[1], &x = a->t[2];
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat_s(pre, RAD_F32, rows, hc) ||
        !mat(x, RAD_BF16, rows, n))
        return R4D_NO(RAD_E_SHAPE);
    const auto al16 = [](const RadTensor& t) {
        return ((uintptr_t)t.data & 15) == 0 && t.stride[0] % 8 == 0;
    };
    if (n % 8 == 0 && al16(h) && al16(x))
        return R4D_EMIT(stream, collapse_vec_kernel,
                        dim3((unsigned)((n / 8 + 255) / 256), grid_rows(rows)), dim3(256), 0,
                        (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,
                        (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data,
                        (long)x.stride[0], (int)n, (int)hc, rows);
    return R4D_EMIT(stream, collapse_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(rows)),
                    dim3(256), 0, (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,
                    (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data, (long)x.stride[0],
                    (int)n, (int)hc, rows);
}

/* ==================================================================== softmax_pool
 * One thread a (entry, channel): the max, the exp sum and the weighted sum over the entry's rows,
 * each in row order; the exp is recomputed for the third pass (the same value). */

__global__ __launch_bounds__(256) void softmax_pool_kernel(const float* st, long sp, uint16_t* out,
                                                           long op, const int32_t* page, long pgs,
                                                           int n, int ratio, long W) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= n) return;
    for (long w = blockIdx.y; w < W; w += gridDim.y) {
        if (page && page[w * pgs] < 0) continue;
        const float* b = st + w * ratio * sp;
        float mx = -INFINITY;
        for (int i = 0; i < ratio; ++i) mx = ds_max(mx, b[i * sp + n + c]);
        float den = 0;
        for (int i = 0; i < ratio; ++i) den += ds_exp(b[i * sp + n + c] - mx);
        float acc = 0;
        for (int i = 0; i < ratio; ++i) acc += ds_exp(b[i * sp + n + c] - mx) / den * b[i * sp + c];
        out[w * op + c] = ds_bf16(acc);
    }
}

int softmax_pool_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 2 || !a->t[0].data || !a->t[1].data) return RAD_E_INVAL;
    const RadTensor &st = a->t[0], &out = a->t[1];
    const RadTensor* page = a->n_t > 2 && a->t[2].data ? &a->t[2] : nullptr;
    const long n = (long)r4d_geti_or(a, "n", 0), ratio = (long)r4d_geti_or(a, "ratio", 0);
    if (n < 1 || n > INT_MAX / 2 || ratio < 1 || ratio > 64 || st.rank != 2 || out.rank != 2 ||
        out.shape[0] < 1 || out.shape[1] != n || st.shape[0] != out.shape[0] * ratio ||
        st.shape[1] != 2 * n || (page && rad_tensor_numel(page) < out.shape[0]))
        return R4D_NO(RAD_E_SHAPE);
    if (st.dtype != RAD_F32 || out.dtype != RAD_BF16 || (page && page->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long W = out.shape[0];
    long pgs = 0;
    if (!mat(st, RAD_F32, W * ratio, 2 * n) || !mat(out, RAD_BF16, W, n) ||
        (page && !lin_step(*page, &pgs)))
        return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, softmax_pool_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(W)),
                    dim3(256), 0, (const float*)st.data, (long)st.stride[0], (uint16_t*)out.data,
                    (long)out.stride[0], page ? (const int32_t*)page->data : nullptr, pgs, (int)n,
                    (int)ratio, W);
}

/* ==================================================================== entry_store
 * One thread an element; a block's n values land at pg * n as libref writes them. */

__global__ __launch_bounds__(256) void entry_store_kernel(const uint16_t* x, long xp,
                                                          const int32_t* page, long pgs,
                                                          uint16_t* cache, long blocks, int n,
                                                          long W) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    for (long w = blockIdx.y; w < W; w += gridDim.y) {
        const long pg = page[w * pgs];
        if (pg < 0 || pg >= blocks) continue;
        cache[pg * n + i] = ds_bf16(ds_bf(x[w * xp + i]));
    }
}

int entry_store_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const long n = (long)r4d_geti_or(a, "n", 0), bs = (long)r4d_geti_or(a, "block_size", 0);
    if (n < 1 || n > INT_MAX || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        rad_tensor_numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] != n)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || page.dtype != RAD_I32 || cache.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long W = x.shape[0];
    long pgs;
    if (!mat(x, RAD_BF16, W, n) || !lin_step(page, &pgs)) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, entry_store_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(W)),
                    dim3(256), 0, (const uint16_t*)x.data, (long)x.stride[0],
                    (const int32_t*)page.data, pgs, (uint16_t*)cache.data, (long)cache.shape[0],
                    (int)n, W);
}

/* ==================================================================== router_scored_topk
 * One wave a row, as r4d_glm.hip's router_sigmoid_topk: lane l holds experts l, l+32, ...; the
 * score is libref's (sigmoid, sqrt(softplus) or the softmax with its sum in expert order), round j
 * takes the largest (rank, ~expert) key below round j-1's and parks it on lane j, so equal ranks go
 * to the lower expert as libref's strict `>` scan takes them. A NaN or -inf rank is never taken.
 * The weights' sum runs in round order. */

enum { kSigmoid = 0, kSqrtSoftplus = 1, kSoftmax = 2 };

__device__ __forceinline__ unsigned long long ds_rank_key(float rank, int e) {
    uint32_t u = __float_as_uint(rank);
    if (u == 0x80000000u) u = 0;
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return rank > -INFINITY ? ((unsigned long long)k << 32) | (uint32_t)~e : 0ull;
}
__device__ __forceinline__ unsigned long long ds_wave_max_u64(unsigned long long b) {
    unsigned long long o;
    o = r4d_xor_lane_u64<16>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<8>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<4>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<2>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<1>(b);
    b = b > o ? b : o;
    return b;
}

template <int OWN, int SCORE>
__global__ __launch_bounds__(128) void router_scored_kernel(const float* logits, long lp,
                                                            const float* bias, long bp, int32_t* ids,
                                                            long ip, float* weights, long wp,
                                                            long m, int ne, int k, int norm,
                                                            float factor) {
    const int lane = threadIdx.x & 31;
    const long r = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    float p[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        p[i] = e < ne ? logits[r * lp + e] : 0.0f;
    }
    if (SCORE == kSoftmax) {
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            if (lane + 32 * i < ne) mx = fmaxf(mx, p[i]);
        mx = r4d_wave_max(mx);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = lane + 32 * i < ne ? ds_exp(p[i] - mx) : 0.0f;
        float den = 0.0f;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            for (int l = 0; l < 32; ++l)
                if (32 * i + l < ne)
                    den += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(p[i]), l));
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = p[i] / den;
    } else if (SCORE == kSigmoid) {
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = 1.0f / (1.0f + ds_exp(-p[i]));
    } else {
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            p[i] = sqrtf(p[i] > 20.0f ? p[i] : ds_log1p(ds_exp(p[i])));
    }
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < ne ? ds_rank_key(p[i] + bias[(long)e * bp], e) : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = ds_wave_max_u64(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[r * ip + lane] = e;
        weights[r * wp + lane] = chosen * mult;
    }
}

template <int SCORE>
int router_emit(RadStream stream, const RadTensor& lg, const RadTensor& bias, const RadTensor& ids,
                const RadTensor& w, int ne, int k, int norm, float factor) {
    const long m = lg.shape[0];
    const dim3 grid((unsigned)((m + 3) / 4));
#define ROUTER(OWN)                                                                                \
    return R4D_EMIT(stream, (router_scored_kernel<OWN, SCORE>), grid, dim3(128), 0,                \
                    (const float*)lg.data, (long)lg.stride[0], (const float*)bias.data,            \
                    (long)bias.stride[0], (int32_t*)ids.data, (long)ids.stride[0],                 \
                    (float*)w.data, (long)w.stride[0], m, ne, k, norm, factor)
    if (ne <= 128) ROUTER(4);
    if (ne <= 256) ROUTER(8);
    if (ne <= 384) ROUTER(12);
    ROUTER(16);
#undef ROUTER
}

int router_scored_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const char* sc = r4d_gets(a, "score");
    if (!sc) return R4D_NO(RAD_E_UNSUPPORTED);
    int score;
    if (!std::strcmp(sc, "sigmoid")) score = kSigmoid;
    else if (!std::strcmp(sc, "sqrtsoftplus")) score = kSqrtSoftplus;
    else if (!std::strcmp(sc, "softmax")) score = kSoftmax;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = (long)r4d_geti_or(a, "n_expert", 0), k = (long)r4d_geti_or(a, "top_k", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    const int norm = r4d_geti_or(a, "norm", 0) != 0 ? 1 : 0;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne || ids.rank != 2 ||
        w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k)
        return R4D_NO(RAD_E_SHAPE);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (lg.stride[1] != 1 || ids.stride[1] != 1 || w.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
    switch (score) {
        case kSigmoid:
            return router_emit<kSigmoid>(stream, lg, bias, ids, w, (int)ne, (int)k, norm, factor);
        case kSqrtSoftplus:
            return router_emit<kSqrtSoftplus>(stream, lg, bias, ids, w, (int)ne, (int)k, norm,
                                              factor);
        default:
            return router_emit<kSoftmax>(stream, lg, bias, ids, w, (int)ne, (int)k, norm, factor);
    }
}

/* ==================================================================== embed_lookup_planes
 * One thread an element of a gathered row, rows over grid.y. The code rows are read in place, at
 * id * rb bytes as libref reads them; only the gathered rows are touched. */

__global__ __launch_bounds__(256) void embed_planes_kernel(
    const int32_t* tok, long ts, const uint8_t* codes, long rb, uint32_t cdt, const void* scale,
    uint32_t sdt, long sp, long ss, const void* table, uint32_t tdt, uint16_t* x, long xp, int E,
    int G, long V, long voff, float mult, long M) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= E) return;
    for (long m = blockIdx.y; m < M; m += gridDim.y) {
        const long id = (long)tok[m * ts] - voff;
        float val = 0.0f;
        if (id >= 0 && id < V) {
            const uint8_t* row = codes + id * rb;
            int code;
            if (cdt == RAD_I8) {
                code = (int)(int8_t)row[i];
            } else if (cdt == RAD_U8) {
                code = (int)row[i];
            } else {
                const int u = (row[i >> 1] >> ((i & 1) * 4)) & 15;
                code = cdt == RAD_I4 ? (u ^ 8) - 8 : u;
            }
            const float v = table ? ds_ld(tdt, table, code) : (float)code;
            val = v * ds_ld(sdt, scale, id * sp + (long)(i / G) * ss) * mult;
        }
        x[m * xp + i] = ds_bf16(val);
    }
}

bool float_plane_dt(uint32_t dt) {
    return dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16 || dt == RAD_F8E4M3 || dt == RAD_E8M0;
}
bool table_dt(uint32_t dt) {
    return dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16 || dt == RAD_F8E4M3 || dt == RAD_I8 ||
           dt == RAD_U8;
}

int embed_planes_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5 || !a->t[0].data || !a->t[1].data || !a->t[2].data || !a->t[4].data)
        return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &cd = a->t[1], &sc = a->t[2], &tb = a->t[3], &x = a->t[4];
    const bool tab = tb.data != nullptr;
    const long E = (long)r4d_geti_or(a, "n_embd", 0), V = (long)r4d_geti_or(a, "n_vocab", 0);
    const long G = (long)r4d_geti_or(a, "group", 0), voff = (long)r4d_geti_or(a, "vocab_offset", 0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    if (E <= 0 || E > INT_MAX || V <= 0 || G <= 0 || E % G || voff < 0 || cd.rank != 2 ||
        cd.shape[0] < V || cd.shape[1] != E || sc.rank != 2 || sc.shape[0] < V ||
        sc.shape[1] != E / G)
        return R4D_NO(RAD_E_SHAPE);
    const uint32_t cdt = cd.dtype;
    const int bits = (cdt == RAD_U4 || cdt == RAD_I4) ? 4 : (cdt == RAD_U8 || cdt == RAD_I8) ? 8 : 0;
    if (!bits) return R4D_NO(RAD_E_DTYPE);
    if (tab && (cdt == RAD_I4 || cdt == RAD_I8 || rad_tensor_numel(&tb) < (int64_t)1 << bits))
        return R4D_NO(RAD_E_DTYPE);
    if (!float_plane_dt(sc.dtype) || (tab && !table_dt(tb.dtype)) || tok.dtype != RAD_I32 ||
        x.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long rb = (long)rad_dtype_bytes(cdt, E);
    const long nx = (long)rad_tensor_numel(&x);
    if (rb <= 0 || nx < E || nx % E) return R4D_NO(RAD_E_SHAPE);
    const long M = nx / E;
    long xp;
    if (x.rank == 2 && x.shape[1] == E) {
        if (x.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
        xp = (long)x.stride[0];
    } else if (rad_tensor_is_contiguous(&x)) {
        xp = E;
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    long ts;
    if (rad_tensor_numel(&tok) < M || !lin_step(tok, &ts)) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, embed_planes_kernel, dim3((unsigned)((E + 255) / 256), grid_rows(M)),
                    dim3(256), 0, (const int32_t*)tok.data, ts, (const uint8_t*)cd.data, rb, cdt,
                    (const void*)sc.data, sc.dtype, (long)sc.stride[0], (long)sc.stride[1],
                    tab ? (const void*)tb.data : nullptr, tab ? tb.dtype : 0u, (uint16_t*)x.data,
                    xp, (int)E, (int)G, V, voff, mult, M);
}

}  // namespace

extern "C" int r4d_rad_token_map(const RadArgs* a, RadStream s) { return token_map_body(a, s); }
extern "C" int r4d_rad_stream_gate_add(const RadArgs* a, RadStream s) {
    return stream_gate_body(a, s);
}
extern "C" int r4d_rad_rope_rows(const RadArgs* a, RadStream s) { return rope_rows_body(a, s); }
extern "C" int r4d_rad_quant_roundtrip(const RadArgs* a, RadStream s) { return quant_rt_body(a, s); }
extern "C" int r4d_rad_mhc_mix(const RadArgs* a, RadStream s) { return mhc_mix_body(a, s); }
/* Partials of every chunk of every row of the band, so the projection spreads over workgroups. */
extern "C" int64_t r4d_rad_mhc_mix_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), n = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M < 1 || n < 1 || hc < 2 || hc > 8 || hc * n > (1LL << 30)) return 0;
    return (int64_t)(M * mix_chunks((long)(hc * n)) * (hc * (hc + 2) + 1) * (long long)sizeof(float));
}
extern "C" int r4d_rad_mhc_collapse(const RadArgs* a, RadStream s) {
    return mhc_collapse_body(a, s);
}
extern "C" int r4d_rad_softmax_pool(const RadArgs* a, RadStream s) {
    return softmax_pool_body(a, s);
}
extern "C" int r4d_rad_entry_store(const RadArgs* a, RadStream s) { return entry_store_body(a, s); }
extern "C" int r4d_rad_router_scored_topk(const RadArgs* a, RadStream s) {
    return router_scored_body(a, s);
}
extern "C" int r4d_rad_embed_lookup_planes(const RadArgs* a, RadStream s) {
    return embed_planes_body(a, s);
}
