/* Independent default-off FP4 aliases. Original producer bodies/exports stay literal.
 * Only nearest-magnitude classification changes. Scale/divide/clamp, norm/rope arithmetic,
 * amax/reductions and BF16 stores are copied from the corresponding original above.
 * SOURCE ONLY until the new alias (including capture/fused/n128/audit) card oracles pass. */
namespace fp4rt = rad_ds41_fp4_runtime;
struct Fp4Local { unsigned long long fast = 0, fallback = 0, captured = 0; };
template <bool Audit, bool Capture>
__device__ __forceinline__ float fp4_classified_q(float v, float scale, Fp4Local& count,
                                                  uint8_t* code) {
    const float clamped = ds_clamp(v / scale, -6.0f, 6.0f);
    rad_ds41_fp4_classifier::Result result;
    const bool fast = rad_ds41_fp4_classifier::classify_bits(__float_as_uint(clamped), &result);
    const float q = fast ? __uint_as_float(result.value_bits) : ds_e2m1_rt(clamped);
    if constexpr (Capture) {
        if (fast) *code = result.code;
        else (void)rad_ds41_kv_codec::code_from_fp32_bits(__float_as_uint(q), code);
    }
    if constexpr (Audit) {
        count.fast += fast ? 1u : 0u;
        count.fallback += fast ? 0u : 1u;
        if constexpr (Capture) count.captured += fast ? 1u : 0u;
    }
    return q;
}
template <int B>
__device__ __forceinline__ void fp4_capture_code(uint8_t code, float scale, uint8_t* row,
                                                 int col, int n, int sc) {
    const uint32_t partner = __shfl_xor((uint32_t)code, 1, 32);
    if (!(col & 1)) row[col / 2] = (uint8_t)(code | (partner << 4));
    if (!(col % B)) row[n / 2 + col / B] = sc == kE4m3 ? (uint8_t)ds_to_e4m3(scale)
                                                       : (uint8_t)(__float_as_uint(scale) >> 23);
}
__device__ __forceinline__ void fp4_checked_add(unsigned long long* p, unsigned long long v,
                                               unsigned long long* overflow) {
    if (!v) return;
    unsigned long long old = atomicCAS(p, 0ull, 0ull);
    for (;;) {
        if (old > ~0ull - v) { atomicExch(overflow, 1ull); return; }
        const unsigned long long seen = atomicCAS(p, old, old + v);
        if (seen == old) return;
        old = seen;
    }
}
__device__ __forceinline__ unsigned long long fp4_shuffle_down(unsigned long long v,unsigned d) {
    const uint32_t lo=__shfl_down((uint32_t)v,d,32),hi=__shfl_down((uint32_t)(v>>32),d,32);
    return (unsigned long long)lo|((unsigned long long)hi<<32);
}
/* One workgroup's local integer totals are folded after its original stores. No global operation
 * per value. Audit=false instantiates no shared storage, reductions, atomics or counter reads. */
template <bool Audit>
__device__ __forceinline__ void fp4_audit_block(unsigned long long* out, Fp4Local count) {
    if constexpr (Audit) {
        __shared__ unsigned long long part[32][3];
        const unsigned lane = threadIdx.x & 31u, wave = threadIdx.x / 32u;
        for (unsigned d = 16; d; d >>= 1) {
            count.fast += fp4_shuffle_down(count.fast, d);
            count.fallback += fp4_shuffle_down(count.fallback, d);
            count.captured += fp4_shuffle_down(count.captured, d);
        }
        if (!lane) { part[wave][0] = count.fast; part[wave][1] = count.fallback; part[wave][2] = count.captured; }
        __syncthreads();
        if (!threadIdx.x) {
            Fp4Local total;
            for (unsigned w = 0; w < blockDim.x / 32u; ++w) {
                total.fast += part[w][0]; total.fallback += part[w][1]; total.captured += part[w][2];
            }
            if (total.fast || total.fallback) {
                fp4_checked_add(out + fp4rt::Fast, total.fast, out + fp4rt::Overflow);
                fp4_checked_add(out + fp4rt::Fallback, total.fallback, out + fp4rt::Overflow);
                fp4_checked_add(out + fp4rt::CapturedFast, total.captured, out + fp4rt::Overflow);
                fp4_checked_add(out + fp4rt::Blocks, 1ull, out + fp4rt::Overflow);
            }
        }
    }
}
int fp4_audit_arg(const RadArgs* a, int base, unsigned long long** out) {
    *out = nullptr;
    if (a->n_t > base + 1) return R4D_NO(RAD_E_INVAL);
    if (a->n_t <= base || !a->t[base].data) return RAD_OK;
    const RadTensor& t = a->t[base];
    if (t.dtype != RAD_I64 || t.rank != 1 || t.shape[0] != fp4rt::kWords ||
        t.stride[0] != 1 || ((uintptr_t)t.data & 7u)) return R4D_NO(RAD_E_SHAPE);
    *out = (unsigned long long*)t.data;
    return RAD_OK;
}

template <int B, bool Capture, bool Audit>
__global__ __launch_bounds__(256) void fp4_quant_group_kernel(uint16_t* x, long xp, int n, long rows,
                                                          int fmt, int sc, KvCapture capture, unsigned long long* audit) {
    Fp4Local count;
    const int col = blockIdx.x * 256 + threadIdx.x;
    const bool in = col < n;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* row = x + r * xp;
        const float v = in ? ds_bf(row[col]) : 0.0f;
        float a = fabsf(v);
        a = a == a ? a : 0.0f;  // std::max(amax, NaN) keeps amax: a NaN never wins
        const float amax = r4d_xor_max<B>(a);
        if (in) {
            const float scale = block_scale(amax, fmt, sc);
            uint8_t code = 0;
            const float q = fp4_classified_q<Audit, Capture>(v, scale, count, &code);
            row[col] = ds_bf16(q * scale);
            if constexpr (Capture) fp4_capture_code<B>(code, scale, capture.p + r * capture.pitch, col, n, sc);
        }
    }
    fp4_audit_block<Audit>(audit, count);
}

template <int B, bool Capture, bool Audit>
__device__ __forceinline__ void fp4_fz_quant_group(const uint16_t* buf, uint16_t* yr, int n,
                                               uint32_t width, int fmt, int sc, uint8_t* packed, Fp4Local& count) {
    for (int base = 0; base < n; base += (int)width) {
        const int col = base + (int)threadIdx.x;
        const bool in = col < n;
        const float v = in ? ds_bf(buf[col]) : 0.0f;
        float a = fabsf(v);
        a = a == a ? a : 0.0f;
        const float amax = r4d_xor_max<B>(a);
        if (in) {
            const float scale = block_scale(amax, fmt, sc);
            uint8_t code = 0;
            const float q = fp4_classified_q<Audit, Capture>(v, scale, count, &code);
            yr[col] = ds_bf16(q * scale);
            if constexpr (Capture) fp4_capture_code<B>(code, scale, packed, col, n, sc);
        }
    }
}

template <bool WF32, uint32_t THR, bool Capture, bool Audit>
__global__ __launch_bounds__(THR) void fp4_norm_rope_rt_kernel(
    const uint16_t* __restrict__ x, uint32_t x_ld, const void* __restrict__ w,
    uint16_t* __restrict__ y, uint32_t y_ld, uint32_t n, float eps, float wadd, float mult,
    uint32_t M, FzRope r, int block, int fmt, int sc, KvCapture capture, unsigned long long* audit) {
    Fp4Local count;
    R4D_DYN_LDS(uint16_t, fz_row);   /* extern __shared__ on the card (r4d_launch.h) */
    __shared__ float lds[THR / kFzWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;   // a row the launch added past the full-card cliff (r4d_cliff_rows)
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;

    // rmsnorm_body's sums; the row and its gains are loaded a batch of kFzCols columns ahead, and
    // stay in registers across the barrier when the row is one batch (n <= kFzCols * THR).
    const bool one = n <= (uint32_t)kFzCols * THR;
    float xv[kFzCols], gv[kFzCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint16_t xb[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            xv[c] = bf16_to_f32(xb[c]);
            if (i0 + (uint32_t)c * THR < n) ss += xv[c] * xv[c];
        }
    }
    ss = fz_block_sum<THR / kFzWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    if (one) {
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) fz_row[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            const float g = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
            fz_row[i] = f32_to_bf16(bf16_to_f32(xr[i]) * s * (g + wadd) * mult);
        }
    }
    __syncthreads();

    const long p = r.pos[(long)m * r.ps];
    if (p >= 0 && p < r.tab_rows)
        for (int q = (int)threadIdx.x; q < r.pairs; q += (int)THR) {
            const int hh = q / r.half, i = q - hh * r.half;
            const long ia = (long)hh * r.hd + r.off + (r.gptj ? 2 * i : i);
            const long ib = r.gptj ? ia + 1 : ia + r.half;
            const float c = r.cs[p * r.cp + (long)i * r.cst];
            const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
            const float va = ds_bf(fz_row[ia]), vb = ds_bf(fz_row[ib]);
            fz_row[ia] = ds_bf16(va * c - vb * sn);
            fz_row[ib] = ds_bf16(va * sn + vb * c);
        }
    __syncthreads();

    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    if (block == 16) fp4_fz_quant_group<16, Capture, Audit>(fz_row, yr, (int)n, THR, fmt, sc,
                            Capture ? capture.p + (long)m * capture.pitch : nullptr, count);
    else fp4_fz_quant_group<32, Capture, Audit>(fz_row, yr, (int)n, THR, fmt, sc,
                            Capture ? capture.p + (long)m * capture.pitch : nullptr, count);
    fp4_audit_block<Audit>(audit, count);
}

template <bool Capture, bool Audit>
__global__ __launch_bounds__(256) void fp4_norm_rope_rt_n128_kernel(
    const uint16_t* __restrict__ x, uint32_t x_ld, const float* __restrict__ w,
    uint16_t* __restrict__ y, uint32_t y_ld, float eps, float wadd, float mult,
    uint32_t M, FzRope r, uint32_t sum_waves, KvCapture capture, unsigned long long* audit) {
    Fp4Local count;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t m = blockIdx.x * 8u + threadIdx.x / kFzWave;
    if (m < M) {
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    float v[4], g[4], ss = 0.0f;
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
        const uint32_t i = lane + c * kFzWave;
        v[c] = bf16_to_f32(xr[i]);
        g[c] = w[i];
        float q = 0.0f;
        q += v[c] * v[c];
        q = r4d_wave_sum(q);
        ss += __shfl(q, 0, 32);
    }
#pragma unroll 1
    for (uint32_t c = 4; c < sum_waves; ++c) ss += 0.0f;
    const float s = rsqrtf(ss / 128.0f + eps);
    uint16_t h[4];
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c)
        h[c] = f32_to_bf16(v[c] * s * (g[c] + wadd) * mult);

    const long p = r.pos[(long)m * r.ps];
    if (p >= 0 && p < r.tab_rows) {
#pragma unroll
        for (uint32_t c = 2; c < 4; ++c) {
            const uint32_t partner = __shfl_xor((uint32_t)h[c], 1, 32);
            const float va = ds_bf((lane & 1u) ? (uint16_t)partner : h[c]);
            const float vb = ds_bf((lane & 1u) ? h[c] : (uint16_t)partner);
            const uint32_t i = ((c - 2u) * kFzWave + lane) / 2u;
            const float co = r.cs[p * r.cp + (long)i * r.cst];
            const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
            h[c] = ds_bf16((lane & 1u) ? va * sn + vb * co : va * co - vb * sn);
        }
    }
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
        const float z = ds_bf(h[c]);
        float a = fabsf(z);
        a = a == a ? a : 0.0f;
        const float amax = r4d_xor_max<32>(a);
        const float scale = block_scale(amax, kFp4, kUe8m0);
        uint8_t code = 0;
        const float q = fp4_classified_q<Audit, Capture>(z, scale, count, &code);
        yr[lane + c * kFzWave] = ds_bf16(q * scale);
        if constexpr (Capture) fp4_capture_code<32>(code, scale, capture.p + (long)m * capture.pitch,
                                                     (int)(lane + c * kFzWave), 128, kUe8m0);
    }
    }
    fp4_audit_block<Audit>(audit, count);
}

template <int B, bool Capture, bool Audit>
__global__ __launch_bounds__(256) void fp4_rope_rt_kernel(uint16_t* x, long xp, int n, long rows,
                                                      FzRope r, int fmt, int sc, KvCapture capture, unsigned long long* audit) {
    Fp4Local count;
    const int col = blockIdx.x * 256 + threadIdx.x;
    const bool in = col < n;
    const int hh = col / r.hd, d = col - hh * r.hd - r.off;
    const bool rot = in && d >= 0 && d < 2 * r.half;
    const int i = d >> 1;
    const bool is_a = (d & 1) == 0;
    for (long m = blockIdx.y; m < rows; m += gridDim.y) {
        uint16_t* row = x + m * xp;
        const uint32_t h = in ? (uint32_t)row[col] : 0u;
        const uint32_t hp = (uint32_t)__shfl_xor((int)h, 1);
        float v = ds_bf((uint16_t)h);
        const long p = r.pos[m * r.ps];
        if (rot && p >= 0 && p < r.tab_rows) {
            const float c = r.cs[p * r.cp + (long)i * r.cst];
            const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
            const float va = is_a ? v : ds_bf((uint16_t)hp), vb = is_a ? ds_bf((uint16_t)hp) : v;
            v = ds_bf(is_a ? ds_bf16(va * c - vb * sn) : ds_bf16(va * sn + vb * c));
        }
        float a = fabsf(v);
        a = a == a ? a : 0.0f;
        const float amax = r4d_xor_max<B>(a);
        if (in) {
            const float scale = block_scale(amax, fmt, sc);
            uint8_t code = 0;
            const float q = fp4_classified_q<Audit, Capture>(v, scale, count, &code);
            row[col] = ds_bf16(q * scale);
            if constexpr (Capture) fp4_capture_code<B>(code, scale, capture.p + m * capture.pitch, col, n, sc);
        }
    }
    fp4_audit_block<Audit>(audit, count);
}

template <bool Audit>
int fp4_quant_body(const RadArgs* a, RadStream stream, bool capture_on, unsigned long long* audit) {
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
    if (!fp4rt::quant_scope(n, block, fmt, sc, capture_on) || x.shape[0] > INT_MAX)
        return quant_rt_body(a, stream, capture_on);
    if (n < 1 || block < 1 || n % block || n > INT_MAX || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != n)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long rows = x.shape[0];
    if (!mat(x, RAD_BF16, rows, n)) return R4D_NO(RAD_E_SHAPE);
    uint16_t* xd = (uint16_t*)x.data;
    const long xp = (long)x.stride[0];
    const dim3 grid((unsigned)((n + 255) / 256), grid_rows(rows));
    KvCapture capture;
    if (capture_on) {
        if (a->n_t < 2) return R4D_NO(RAD_E_INVAL);
        const int rc = kv_capture_arg(a, a->t[1], rows, n, block, fmt, sc, &capture);
        if (rc < 0) return rc;
        if (block == 16) return R4D_EMIT(stream, (fp4_quant_group_kernel<16, true, Audit>), grid, dim3(256), 0,
                                        xd, xp, (int)n, rows, fmt, sc, capture, audit);
        return R4D_EMIT(stream, (fp4_quant_group_kernel<32, true, Audit>), grid, dim3(256), 0,
                         xd, xp, (int)n, rows, fmt, sc, capture, audit);
    }
    if (block == 16) return R4D_EMIT(stream, (fp4_quant_group_kernel<16, false, Audit>), grid, dim3(256), 0,
                                     xd, xp, (int)n, rows, fmt, sc, capture, audit);
    return R4D_EMIT(stream, (fp4_quant_group_kernel<32, false, Audit>), grid, dim3(256), 0,
                    xd, xp, (int)n, rows, fmt, sc, capture, audit);
}

template <bool Audit>
int fp4_norm_body(const RadArgs* a, RadStream stream, int n128, bool capture_on, unsigned long long* audit) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2], &cs = a->t[3], &pos = a->t[4];
    if (!x.data || !w.data || !y.data || !cs.data || !pos.data) return R4D_NO(RAD_E_INVAL);
    if (x.dtype != RAD_BF16 || y.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(&w)) return R4D_NO(RAD_E_STRIDE);
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || n > kFzMaxN || r4d_numel(&w) < n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(&x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || M > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    FzRope r;
    const int rc = fz_rope_args(a, y, cs, pos, (long)M, &r);
    if (rc != RAD_OK) return R4D_NO(rc);
    if ((long long)r.hd * (r.pairs / r.half) != n) return R4D_NO(RAD_E_SHAPE);
    const long long block = r4d_geti_or(a, "block", 0);
    int fmt, sc;
    if (!fz_quant_fmt(a, &fmt, &sc)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (block < 1 || n % block) return R4D_NO(RAD_E_SHAPE);
    if (n != 128 || !fp4rt::quant_scope(n, block, fmt, sc, capture_on) ||
        !fp4rt::rope_scope(n, r.hd, r.half, r.off, r.pairs, r.gptj != 0, r.sign))
        return norm_rope_rt_body(a, stream, n128, capture_on);
    KvCapture capture;
    if (capture_on) {
        if (a->n_t < 6) return R4D_NO(RAD_E_INVAL);
        const int pr = kv_capture_arg(a, a->t[5], (long)M, (long)n, (long)block, fmt, sc, &capture);
        if (pr < 0) return pr;
    }
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    const int thr = r4d_act_threads((int)M, (int)n);
    const bool short_key = n == 128 && w.dtype == RAD_F32 && r.hd == 128 && r.half == 32 &&
                           r.off == 64 && r.pairs == 32 && r.gptj && r.sign == 1.0f &&
                           block == 32 && fmt == kFp4 && sc == kUe8m0;
    /* Strictly default off, read once. Forced 0/1 is only the checker export below; a forced 1
     * outside this exact site refuses rather than silently checking the old kernel twice. */
    static const bool short_key_on = [] {
        const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_N128");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    if (n128 == 1 && !short_key) return R4D_NO(RAD_E_UNSUPPORTED);
    if (short_key && (n128 == 1 || (n128 < 0 && short_key_on))) {
#define FZ_N128(PK) R4D_EMIT(stream, (fp4_norm_rope_rt_n128_kernel<PK, Audit>),                    \
                         dim3(r4d_cliff_rows(((uint32_t)M + 7u) / 8u, 1, 8)), dim3(256), 0, \
                         (const uint16_t*)x.data, (uint32_t)x_ld, (const float*)w.data, \
                         (uint16_t*)y.data, (uint32_t)y.stride[0], eps, wadd, mult, (uint32_t)M, \
                         r, (uint32_t)thr / kFzWave, capture, audit)
        const int launch = capture_on ? FZ_N128(true) : FZ_N128(false);
#undef FZ_N128
        /* Proof for a model arm, only after its normal dispatcher queued the fast kernel.
         * Forced checker calls cannot produce it. After each rank's first use only a relaxed load
         * remains; disabled, ineligible and failed launches never touch this state or perform I/O. */
        if (n128 < 0 && launch == RAD_OK && a->rank >= 0 && a->rank < a->world_size && a->rank < 64) {
            static std::atomic<uint64_t> reported{0};
            const uint64_t bit = uint64_t(1) << (unsigned)a->rank;
            if (!(reported.load(std::memory_order_relaxed) & bit) &&
                !(reported.fetch_or(bit, std::memory_order_relaxed) & bit)) {
                std::fprintf(stderr, "[r4d] ds41 fp4_classify n128 active: normal dispatch rank=%d world=%d n=128 M=%lld old_thr=%d\n",
                             a->rank, a->world_size, M, thr);
                std::fflush(stderr);
            }
        }
        return launch;
    }
    const uint32_t lds = (uint32_t)(n * sizeof(uint16_t));
#define FZ_NRQ(WF, T, PK)                                                                         \
    R4D_EMIT(stream, (fp4_norm_rope_rt_kernel<WF, T, PK, Audit>), dim3(r4d_cliff_rows((uint32_t)M, 1, T / kFzWave)), \
             dim3(T), lds, (const uint16_t*)x.data, (uint32_t)x_ld, (const void*)w.data,          \
             (uint16_t*)y.data, (uint32_t)y.stride[0], (uint32_t)n, eps, wadd, mult, (uint32_t)M, \
             r, (int)block, fmt, sc, capture, audit)
#define FZ_NRQ_W(WF, PK)                                                                          \
    return thr == 1024 ? FZ_NRQ(WF, 1024, PK) : thr == 512 ? FZ_NRQ(WF, 512, PK) : FZ_NRQ(WF, 256, PK)
    if (capture_on) {
        if (w.dtype == RAD_F32) FZ_NRQ_W(true, true);
        FZ_NRQ_W(false, true);
    }
    if (w.dtype == RAD_F32) FZ_NRQ_W(true, false);
    FZ_NRQ_W(false, false);
#undef FZ_NRQ_W
#undef FZ_NRQ
}

template <bool Audit>
int fp4_rope_body(const RadArgs* a, RadStream stream, bool capture_on, unsigned long long* audit) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &cs = a->t[1], &pos = a->t[2];
    if (x.rank != 2 || x.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long M = (long)x.shape[0];
    FzRope r;
    const int rc = fz_rope_args(a, x, cs, pos, M, &r);
    if (rc != RAD_OK) return R4D_NO(rc);
    const long n = (long)r.hd * (r.pairs / r.half);
    const long block = (long)r4d_geti_or(a, "block", 0);
    int fmt, sc;
    if (!fz_quant_fmt(a, &fmt, &sc)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (block < 1 || n % block) return R4D_NO(RAD_E_SHAPE);
    if (M > INT_MAX || !fp4rt::quant_scope(n, block, fmt, sc, capture_on) ||
        !fp4rt::rope_scope(n, r.hd, r.half, r.off, r.pairs, r.gptj != 0, r.sign))
        return rope_rt_body(a, stream, capture_on);
    KvCapture capture;
    if (capture_on) {
        if (a->n_t < 4) return R4D_NO(RAD_E_INVAL);
        const int pr = kv_capture_arg(a, a->t[3], M, n, block, fmt, sc, &capture);
        if (pr < 0) return pr;
    }
    const dim3 grid((unsigned)((n + 255) / 256), grid_rows(M));
    uint16_t* xd = (uint16_t*)x.data;
    const long xp = (long)x.stride[0];
    if (capture_on) {
        if (block == 16) return R4D_EMIT(stream, (fp4_rope_rt_kernel<16, true, Audit>), grid, dim3(256), 0,
                                         xd, xp, (int)n, M, r, fmt, sc, capture, audit);
        return R4D_EMIT(stream, (fp4_rope_rt_kernel<32, true, Audit>), grid, dim3(256), 0,
                         xd, xp, (int)n, M, r, fmt, sc, capture, audit);
    }
    if (block == 16) return R4D_EMIT(stream, (fp4_rope_rt_kernel<16, false, Audit>), grid, dim3(256), 0,
                                     xd, xp, (int)n, M, r, fmt, sc, capture, audit);
    return R4D_EMIT(stream, (fp4_rope_rt_kernel<32, false, Audit>), grid, dim3(256), 0,
                    xd, xp, (int)n, M, r, fmt, sc, capture, audit);
}

int fp4_quant_dispatch(const RadArgs* a, RadStream s, bool capture) {
    if (!a) return RAD_E_INVAL;
    unsigned long long* audit = nullptr;
    const int rc = fp4_audit_arg(a, capture ? 2 : 1, &audit); if (rc < 0) return rc;
    const int at=fp4rt::slot(capture?fp4rt::QuantCapture:fp4rt::Quant,r4d_geti_or(a,"n",0));
    audit=audit&&at>=0?audit+at*fp4rt::FieldCount:nullptr;
    return audit ? fp4_quant_body<true>(a,s,capture,audit) : fp4_quant_body<false>(a,s,capture,nullptr);
}
int fp4_norm_dispatch(const RadArgs* a, RadStream s, bool capture, int n128 = -1) {
    if (!a) return RAD_E_INVAL;
    unsigned long long* audit = nullptr;
    const int rc = fp4_audit_arg(a, capture ? 6 : 5, &audit); if (rc < 0) return rc;
    const int at=fp4rt::slot(capture?fp4rt::NormCapture:fp4rt::Norm,r4d_geti_or(a,"n",0));
    audit=audit&&at>=0?audit+at*fp4rt::FieldCount:nullptr;
    return audit ? fp4_norm_body<true>(a,s,n128,capture,audit) : fp4_norm_body<false>(a,s,n128,capture,nullptr);
}
int fp4_rope_dispatch(const RadArgs* a, RadStream s, bool capture) {
    if (!a) return RAD_E_INVAL;
    unsigned long long* audit = nullptr;
    const int rc = fp4_audit_arg(a, capture ? 4 : 3, &audit); if (rc < 0) return rc;
    const long long nh=r4d_geti_or(a,"n_head",0),hd=r4d_geti_or(a,"head_dim",0);
    const long long n=nh>0&&hd>0&&nh<=4096/hd?nh*hd:0;
    const int at=fp4rt::slot(capture?fp4rt::RopeCapture:fp4rt::Rope,n);
    audit=audit&&at>=0?audit+at*fp4rt::FieldCount:nullptr;
    return audit ? fp4_rope_body<true>(a,s,capture,audit) : fp4_rope_body<false>(a,s,capture,nullptr);
}
