/* The exact n128 fused index-key fast path against the unchanged fused workgroup kernel.
 * The same cases run on the SIMT emulator and, with DS41_N128_DEVICE, a loaded libr4d on a
 * card. Card runs belong ONLY in the research two-lock QFN stop/restore window.
 * No libref/unfused norm oracle: upstream's standalone norm changed its partition; the current
 * fused kernel is the byte contract here. Both forms must also preserve every padding byte.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef DS41_N128_DEVICE
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#else
extern const char* emu_last_kernel;
#endif

namespace {

using Launch = int (*)(const RadArgs*, RadStream);
using Check = int (*)(const RadArgs*, RadStream, int);
Launch normal;
Check check;
#ifdef DS41_N128_DEVICE
int (*record)(const void**, int);
const void* last_dispatch;
#endif

bool enabled() {
    const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_N128");
    return e && !std::strcmp(e, "1");
}

uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
float bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
RadParam pi(const char* k, int v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadTensor tensor(void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int i = 0; for (int64_t n : shape) t.shape[i++] = n;
    int64_t stride = 1;
    for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = stride; stride *= t.shape[k]; }
    return t;
}

struct Case {
    static constexpr int n = 128, xp = 133, yp = 135, guard = 64, P = 5, rot = 64;
    int M, profile;
    std::vector<uint16_t> x, y;
    std::vector<float> w, cs;
    std::vector<int32_t> pos;
    std::vector<RadParam> p;
    RadTensor t[5];
    RadArgs a{};

    Case(int rows, int kind) : M(rows), profile(kind),
        x(guard + (size_t)(rows + 2) * xp + guard, 0x7fc1),
        y(guard + (size_t)(rows + 2) * yp + guard, 0xa5a5), w(n),
        cs((size_t)P * (rot * 2 + 3), -7.f), pos((size_t)rows * 2, -77) {
        /* Input, output, positions and the rope table all have independent nontrivial pitches. */
        for (int m = 0; m < M; ++m) {
            pos[(size_t)m * 2] = (m % 7) - 1;  // -1, 0..4, P (both invalid positions skip rope)
            for (int j = 0; j < n; ++j) {
                uint16_t h = rad_f32_to_bf16((float)((int)(mix((uint32_t)(m * n + j)) % 2001) - 1000) / 512.f);
                if (profile == 0) {
                    if (m % 11 == 0) h = 0;                      // zero row
                    if (m % 11 == 1) h = j & 1 ? 0x8000 : 0;     // signed zeros
                    if (m % 11 == 2) h = (uint16_t)(1 + j % 127); // bf16 subnormals
                    if (m % 11 == 3) h = (uint16_t)(0x7f00 + j % 128); // square overflow
                }
                if (profile == 1 && j == (m * 17) % n) {
                    const uint16_t special[] = { 0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x7f81, 0xff83 };
                    h = special[m % 6];
                }
                if (profile == 1 && j == (m * 17 + 33) % n)
                    h = (uint16_t)(0x7f85u + (uint32_t)(m % 29)); // competing NaN payload, next chunk
                if (profile == 2 || profile >= 4) h = 0x3f80; // factor exactly 1 with eps0
                x[guard + (size_t)m * xp + j] = h;
            }
        }
        for (int j = 0; j < n; ++j) {
            w[(size_t)j] = bits(0x3f000000u + mix((uint32_t)j + 77u) % 0x01000000u);
            if (profile == 2) {
                const uint16_t midpoint[] = { 0x3e80, 0x3f40, 0x3fa0, 0x3fe0, 0x4020, 0x4060, 0x40a0 };
                const uint16_t h = (uint16_t)(midpoint[(j / 3) % 7] + j % 3 - 1);
                w[(size_t)j] = rad_bf16_to_f32(h) * (j & 1 ? -1.f : 1.f);
                if ((j & 31) == 31) w[(size_t)j] = 6.f; // UE8M0 scale1 in each FP4 group
            }
            if (profile == 3) {
                const uint32_t edge[] = { 0, 0x80000000u, 0x3f807fffu, 0x3f808000u, 0x3f808001u,
                                         0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc54321u,
                                         0x7f812345u, 0xff812346u, 1, 0x007fffffu };
                w[(size_t)j] = bits(edge[j % (sizeof edge / sizeof edge[0])]);
            }
            if (profile == 4) {
                /* The norm's bf16 rounding straddles an FP4 midpoint, so the final result
                 * observes the original store's tie-to-even decision, including its low bit. */
                const uint32_t high[] = { 0x3e80, 0x3f40, 0x3fa0, 0x3fe0, 0x4020, 0x4060, 0x40a0 };
                const uint32_t low[] = { 0x7fff, 0x8000, 0x8001 };
                w[(size_t)j] = bits((high[(j / 3) % 7] << 16) | low[j % 3] | (j & 1 ? 0x80000000u : 0));
                if ((j & 31) == 31) w[(size_t)j] = 6.f;
            }
            if (profile == 5) {
                uint16_t h;
                if (j < 32) h = j & 1 ? 0x8000 : 0; // zero-group scale floor
                else if (j < 64) h = (uint16_t)(0x01be + j % 3); // around 6 * 2^-126
                else if (j < 96) h = (uint16_t)(0x40be + j % 3); // around UE8M0 scale1
                else h = (uint16_t)(0x7f7d + j % 3); // maximum finite bf16
                w[(size_t)j] = rad_bf16_to_f32(h);
            }
        }
        for (int q = 0; q < P; ++q)
            for (int j = 0; j < rot / 2; ++j) {
                const double angle = q * std::pow(10000.0, -2.0 * j / rot);
                const bool identity = profile == 2 || profile >= 4;
                cs[(size_t)q * (rot * 2 + 3) + j * 2] = identity ? 1.f : (float)std::cos(angle);
                cs[(size_t)q * (rot * 2 + 3) + (rot / 2 + j) * 2] = identity ? 0.f : (float)std::sin(angle);
            }
        if (profile == 3) {
            cs[(rot * 2 + 3) + 2] = bits(0x7fc12345u);
            cs[2 * (rot * 2 + 3) + 4] = bits(0xff812345u);
        }
        p = { pi("n", n), pf("eps", profile == 2 || profile >= 4 ? 0 : 1e-6), pi("n_head", 1), pi("head_dim", n),
              pi("rot", rot), pi("off", n - rot), ps("mode", "gptj"), pi("inverse", 0),
              pi("block", 32), ps("format", "fp4_e2m1"), ps("scale", "ue8m0"), ps("dtype", "bf16"),
              pf("wadd", profile == 3 ? -0.125 : 0), pf("mult", profile == 3 ? -1.25 : 1), pi("M", M) };
        t[0] = tensor(x.data() + guard, RAD_BF16, { M, n }); t[0].stride[0] = xp;
        t[1] = tensor(w.data(), RAD_F32, { n });
        t[2] = tensor(y.data() + guard, RAD_BF16, { M, n }); t[2].stride[0] = yp;
        t[3] = tensor(cs.data(), RAD_F32, { P, rot });
        t[3].stride[0] = rot * 2 + 3; t[3].stride[1] = 2;
        t[4] = tensor(pos.data(), RAD_I32, { M }); t[4].stride[0] = 2;
        a.t = t; a.n_t = 5; a.p = p.data(); a.n_p = (int)p.size();
    }
    bool guards(const std::vector<uint16_t>& out) const {
        for (size_t i = 0; i < out.size(); ++i) {
            const bool output = i >= guard && i < guard + (size_t)M * yp && (i - guard) % yp < n;
            if (!output && out[i] != 0xa5a5) return false;
        }
        return true;
    }
};

#ifdef DS41_N128_DEVICE
void hc(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e)); std::exit(2); }
}
#endif

int run(Case& c, int form, std::vector<uint16_t>* out, bool eligible = true) {
    std::fill(c.y.begin(), c.y.end(), 0xa5a5);
#ifdef DS41_N128_DEVICE
    void* d[5]{};
    void* h[5] = { c.x.data(), c.w.data(), c.y.data(), c.cs.data(), c.pos.data() };
    const size_t bytes[5] = { c.x.size() * 2, c.w.size() * 4, c.y.size() * 2,
                              c.cs.size() * 4, c.pos.size() * 4 };
    RadTensor t[5]; std::memcpy(t, c.t, sizeof t);
    for (int i = 0; i < 5; ++i) {
        hc(hipMalloc(&d[i], bytes[i]));
        hc(hipMemcpy(d[i], h[i], bytes[i], hipMemcpyHostToDevice));
        const size_t off = (const char*)c.t[i].data - (const char*)h[i];
        t[i].data = (char*)d[i] + off;
    }
    RadArgs a = c.a; a.t = t;
    const void* dispatched[2]{};
    record(dispatched, 2);
    const int rc = form < 0 ? normal(&a, nullptr) : check(&a, nullptr, form);
    const int count = record(nullptr, 0);
    last_dispatch = count == 1 ? dispatched[0] : nullptr;
    hc(hipDeviceSynchronize());
    hc(hipMemcpy(c.y.data(), d[2], bytes[2], hipMemcpyDeviceToHost));
    for (void* p : d) hc(hipFree(p));
    if ((rc == RAD_OK && !last_dispatch) || (rc != RAD_OK && count != 0)) {
        std::printf("FAIL: n128 form%d dispatched %d kernels, rc%d\n", form, count, rc);
        return RAD_E_DEVICE;
    }
    (void)eligible;
#else
    const int rc = form < 0 ? normal(&c.a, nullptr) : check(&c.a, nullptr, form);
    if (rc == RAD_OK) {
        const bool want = eligible && (form == 1 || (form < 0 && enabled()));
        if ((std::strstr(emu_last_kernel, "norm_rope_rt_n128_kernel") != nullptr) != want) {
            std::printf("FAIL: wrong dispatch form%d: %s\n", form, emu_last_kernel);
            return RAD_E_DEVICE;
        }
    }
#endif
    *out = c.y;
    return rc;
}

int compare(Case& c) {
    std::vector<uint16_t> base, fast, selected;
    const int a = run(c, 0, &base);
#ifdef DS41_N128_DEVICE
    const void* base_fn = last_dispatch;
#endif
    const int b = run(c, 1, &fast);
#ifdef DS41_N128_DEVICE
    const void* fast_fn = last_dispatch;
#endif
    const int d = run(c, -1, &selected);
    bool dispatch = true;
#ifdef DS41_N128_DEVICE
    dispatch = base_fn && fast_fn && base_fn != fast_fn && last_dispatch == (enabled() ? fast_fn : base_fn);
#endif
    const bool same = a == RAD_OK && b == RAD_OK && d == RAD_OK && base == fast && base == selected &&
                      dispatch && c.guards(base) && c.guards(fast) && c.guards(selected);
    size_t first = 0;
    if (a == RAD_OK && b == RAD_OK) while (first < base.size() && base[first] == fast[first]) ++first;
    std::printf("%s: n128 M%d profile%d every byte, guards and dispatch (rc%d/%d/%d, first difference %zu)\n",
                same ? "PASS" : "FAIL", c.M, c.profile, a, b, d, first);
    return same ? 0 : 1;
}

int refusals() {
    int failed = 0;
    /* All these remain legal for the old fused row; forced fast must specifically refuse them.
     * Each fresh case makes the altered premise independent of all the others. */
    for (int k = 0; k < 10; ++k) {
        Case c(1, 0);
        switch (k) {
        case 0: c.p[4].ival = 32; c.p[5].ival = 96; c.t[3].shape[1] = 32; break;
        case 1: c.p[5].ival = 0; break;
        case 2: c.p[6].sval = "neox"; break;
        case 3: c.p[7].ival = 1; break;
        case 4: c.p[8].ival = 16; break;
        case 5: c.p[9].sval = "fp8_e4m3"; break;
        case 6: c.p[10].sval = "e4m3"; break;
        case 7: c.t[1].dtype = RAD_BF16; break;
        case 8: c.p[2].ival = 2; c.p[3].ival = 64; c.p[4].ival = 32; c.p[5].ival = 32;
                c.t[3].shape[1] = 32; break;
        case 9: c.p[0].ival = 64; c.p[3].ival = 64; c.p[4].ival = 32; c.p[5].ival = 32;
                c.t[0].shape[1] = c.t[2].shape[1] = c.t[1].shape[0] = 64;
                c.t[3].shape[1] = 32; break;
        }
        std::vector<uint16_t> got;
        const int rc = run(c, 1, &got);
        const bool untouched = std::all_of(got.begin(), got.end(), [](uint16_t h) { return h == 0xa5a5; });
        std::vector<uint16_t> base, fallback;
        const int b = run(c, 0, &base, false);
#ifdef DS41_N128_DEVICE
        const void* base_fn = last_dispatch;
#endif
        const int f = run(c, -1, &fallback, false);
        bool dispatch = true;
#ifdef DS41_N128_DEVICE
        dispatch = base_fn && last_dispatch == base_fn;
#endif
        const bool ok = rc == RAD_E_UNSUPPORTED && untouched && b == RAD_OK && f == RAD_OK &&
                        base == fallback && dispatch && c.guards(base) && c.guards(fallback);
        std::printf("%s: n128 eligibility near-miss%d refused before writing; unchanged fallback, rc%d/%d/%d\n",
                    ok ? "PASS" : "FAIL", k, rc, b, f);
        failed += !ok;
    }
    Case c(1, 0);
    std::vector<uint16_t> got;
    const int rc = run(c, 2, &got);
    const bool ok = rc == RAD_E_INVAL &&
                    std::all_of(got.begin(), got.end(), [](uint16_t h) { return h == 0xa5a5; });
    std::printf("%s: n128 invalid checker form refused, rc%d\n", ok ? "PASS" : "FAIL", rc);
    return failed + !ok;
}

} // namespace

int main(int argc, char** argv) {
#ifdef DS41_N128_DEVICE
    if (argc != 2) { std::fprintf(stderr, "usage: %s LIBR4D.so (inside GPU window only)\n", argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    normal = (Launch)dlsym(h, "r4d_rad_rmsnorm_rope_roundtrip");
    check = (Check)dlsym(h, "r4d_rad_rmsnorm_rope_roundtrip_n128_check");
    record = (int (*)(const void**, int))dlsym(h, "r4d_record_dispatches");
    if (!normal || !check || !record) { std::fprintf(stderr, "FAIL: n128 checker exports absent\n"); return 2; }
#else
    (void)argc; (void)argv;
    normal = r4d_rad_rmsnorm_rope_roundtrip;
    check = r4d_rad_rmsnorm_rope_roundtrip_n128_check;
#endif
    int failed = 0, cases = 0, max_m = 0;
    auto positive = [&](int m, int profile) {
        Case c(m, profile);
        failed += compare(c);
        ++cases;
        max_m = std::max(max_m, m);
    };
    for (int m : { 1, 7, 8, 9, 16, 32, 33, 72, 73 })
        positive(m, 0); // THR1024/512/256 without the LCP pin
#ifdef DS41_N128_DEVICE
    for (int m : { 255, 256, 1024, 2048, 4096 }) positive(m, 0);
#endif
    for (int profile : { 1, 2, 3, 4, 5 })
        for (int m : { 17, 33, 73 }) positive(m, profile);
    failed += refusals();
    std::printf("%s: ds41 n128 wave norm matches unchanged fused norm, all output/guard bytes; "
                "exceptional and rounding cases, eligibility refusals; through M%d (%d positive cases)\n",
                failed ? "FAIL" : "PASS", max_m, cases);
    return failed ? 1 : 0;
}
