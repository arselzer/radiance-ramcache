/* Strict uint16 DS41 KV codec oracle. Run ONLY in the ordinary two-lock stopped-QFN window.
 * No arguments: standalone code/scale-domain proof against the CURRENT base's scalar arithmetic.
 * LIBR4D.so: also compare original roundtrip/store with capture+packed-store exports on actual
 * bf16 producer inputs, pitched source rows, skipped pages and zero-initialized physical pages.
 * NaN products are evaluated on THIS card; no CPU NaN payload assumption or float equality.
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rad_plugin.h"
#include "../../arch/common/rad_ds41_kv_codec.h"

namespace kc = rad_ds41_kv_codec;
#define HC(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "FAIL %s: %s\n", #x, hipGetErrorString(e)); std::exit(2); } } while (0)
namespace {
/* Independent transcription of current r4d_ds41_elem.hip: ds_bf16, ds_e4m3 and the exact FP4
 * magnitudes ds_e2m1_rt can return. Deliberately does not call the codec's decoder/bit tables. */
__device__ uint16_t old_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ float old_e4m3(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31, e = (b >> 3) & 15;
    uint32_t m = b & 7, u = 0;
    if (!e) {
        if (!m) u = s;
        else {
            int sh = 0; while (!(m & 8)) { m <<= 1; ++sh; }
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | ((m & 7) << 20);
        }
    } else if (e == 15 && m == 7) u = s | 0x7FC00000u;
    else u = s | ((e - 7 + 127) << 23) | (m << 20);
    return __uint_as_float(u);
}
__device__ float old_q(uint8_t c) {
    const float mag[8] = { 0, .5f, 1, 1.5f, 2, 3, 4, 6 };
    return c & 8 ? -mag[c & 7] : mag[c & 7];
}
struct Desc { uint8_t kind, source_scale; };
struct Report { unsigned cases = 0, normal = 0, pair = 0, ftz = 0, raw = 0, bad = 0; uint16_t half_pos = 0, half_neg = 0; };
__global__ void domain_oracle(const Desc* desc, int count, Report* report) {
    const int at = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (at >= count) return;
    const auto k = (kc::Kind)desc[at].kind;
    const uint8_t source_scale = desc[at].source_scale;
    const int G = kc::group_size(k), bytes = G / 2;
    uint8_t codes[32], guarded[32], row[17];
    uint16_t current[32];
    for (int i = 0; i < 32; ++i) guarded[i] = 0xA5;
    const float scale = k == kc::Kind::Fp4E4M3G16 ? old_e4m3(source_scale)
                                                : __uint_as_float((uint32_t)source_scale << 23);
    for (int i = 0; i < G; ++i) {
        // Original scale255 can ONLY produce q=+0/-0, after NaN is mapped to FP4 +0.
        codes[i] = source_scale == 0 ? 0 : k == kc::Kind::Fp4Pow2G32 && source_scale == 255
                    ? (uint8_t)((i & 1) ? 8 : 0) : (uint8_t)(i & 15);
        const float product = old_q(codes[i]) * scale;
        current[i] = old_bf16(product);
    }
    if (k == kc::Kind::Fp4Pow2G32 && source_scale == 1) {
        report->half_pos = current[1]; report->half_neg = current[9];
    }
    uint8_t stored = 77;
    const auto result = kc::pack_captured_group(k, source_scale, codes, current, guarded + 8, &stored);
    atomicAdd(&report->cases, 1);
    if (result != kc::PackResult::Exact) { atomicAdd(&report->raw, 1); return; }
    atomicAdd(stored == 0 ? &report->pair : stored == 254 ? &report->ftz : &report->normal, 1);
    unsigned bad = 0;
    for (int i = 0; i < 32; ++i) if (i < 8 || i >= 8 + bytes) bad += guarded[i] != 0xA5;
    for (int i = 0; i < bytes; ++i) row[i] = guarded[8 + i];
    row[bytes] = stored;
    for (int i = 0; i < G; ++i) {
        uint16_t got = 0xBEEF;
        bad += !kc::decode_group_value(k, stored, guarded + 8, i, &got) || got != current[i];
        bad += kc::decode_row_bits(k, row, G, i) != current[i];
    }
    if (!source_scale) for (int i = 0; i <= bytes; ++i) bad += row[i] != 0;
    atomicAdd(&report->bad, bad);
}

using Launch = int (*)(const RadArgs*, RadStream);
RadTensor tensor(void* data, uint32_t dtype, std::initializer_list<int64_t> shape) {
    RadTensor t{}; t.data = data; t.dtype = dtype; t.rank = (uint32_t)shape.size();
    int i = 0; for (int64_t n : shape) t.shape[i++] = n;
    int64_t stride = 1;
    for (i = (int)t.rank - 1; i >= 0; --i) { t.stride[i] = stride; stride *= t.shape[i]; }
    return t;
}
RadParam pi(const char* key, int64_t n) { RadParam p{}; p.key = key; p.kind = RAD_P_INT; p.ival = n; return p; }
RadParam ps(const char* key, const char* s) { RadParam p{}; p.key = key; p.kind = RAD_P_STR; p.sval = s; return p; }
int call(Launch launch, std::vector<RadTensor>& tensors, std::vector<RadParam>& params) {
    RadArgs a{}; a.t = tensors.data(); a.n_t = (int)tensors.size(); a.p = params.data(); a.n_p = (int)params.size();
    a.world_size = 1;
    return launch(&a, nullptr);
}
uint16_t input_bits(int64_t row, int64_t col, int G) {
    // Explicit block amax floors, all-NaN, Inf, signed zero and maximal finite exponent cases.
    // The separate M512 producer run below supplies every BF16 bit pattern in original order.
    const int mode = (int)((row + col / G) % 8);
    const int j = (int)(col % G);
    if (mode == 0) return j & 1 ? 0x8000 : 0;
    if (mode == 1) return (uint16_t)((j & 1 ? 0x8000 : 0) | (j + 1));
    if (mode == 2) return (uint16_t)((j & 1 ? 0xFF81 : 0x7F81) + j); // signaling NaN payloads
    if (mode == 3) return j == 0 ? 0x7F80 : j == 1 ? 0xFF80 : (uint16_t)(0x3F00 + 17 * j);
    if (mode == 4) return (uint16_t)((j & 1 ? 0xFF00 : 0x7F00) | (j * 11 & 127));
    if (mode == 5) return (uint16_t)((j & 1 ? 0x8000 : 0) | (j % 4 == 0 ? 0x0140 : j % 4 == 1 ? 0x0040 : 0x0080));
    return (uint16_t)(row * 40503u + col * 23063u);
}
bool producer_oracle(const char* path) {
    void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "FAIL dlopen: %s\n", dlerror()); return false; }
    const auto plain_rt = (Launch)dlsym(lib, "r4d_rad_quant_roundtrip");
    const auto packed_rt = (Launch)dlsym(lib, "r4d_rad_quant_roundtrip_kvpack");
    const auto plain_store = (Launch)dlsym(lib, "r4d_rad_entry_store");
    const auto packed_store = (Launch)dlsym(lib, "r4d_rad_entry_store_kvpack");
    if (!plain_rt || !packed_rt || !plain_store || !packed_store) {
        std::fprintf(stderr, "FAIL producer/store export missing (no silent standalone fallback)\n"); dlclose(lib); return false;
    }
    size_t bad = 0, cases = 0;
    for (uint32_t kind : { 1u, 2u }) for (int M : { 1, 7, 33, 257, 512 }) for (int bs : { 1, 2 }) {
        const int n = kind == 1 ? 512 : 128, G = kc::group_size(kind), pitch = n + 16, blocks = M + 3;
        const int64_t rb = kc::row_bytes(kind, n);
        constexpr int guard = 16;
        const size_t nx = (size_t)(2 * guard + M * pitch), nc = (size_t)(2 * guard + blocks * n);
        const size_t np = (size_t)(2 * guard + M * rb), nk = (size_t)(2 * guard + blocks * rb);
        std::vector<uint16_t> input(nx, 0x5AA5), ref(nx), actual(nx), cache(nc, 0), got_cache(nc), decoded((size_t)n);
        std::fill(cache.begin(), cache.begin() + guard, 0x5AA5); std::fill(cache.end() - guard, cache.end(), 0x5AA5);
        for (int r = 0; r < M; ++r) for (int c = 0; c < n; ++c)
            input[(size_t)(guard + r * pitch + c)] = M == 512 ? (uint16_t)(r * n + c) : input_bits(r, c, G);
        std::vector<uint8_t> capture(np, 0xA5), packed(nk, 0), got_capture(np), got_packed(nk);
        std::fill(packed.begin(), packed.begin() + guard, 0xA5); std::fill(packed.end() - guard, packed.end(), 0xA5);
        std::vector<int32_t> page((size_t)M);
        for (int r = 0; r < M; ++r) page[(size_t)r] = M - r + 1;
        page[0] = -1; if (M > 2) page[(size_t)M - 1] = blocks;
        void *a = nullptr, *b = nullptr, *cap = nullptr, *old_cache = nullptr, *new_cache = nullptr, *pg = nullptr;
        HC(hipMalloc(&a, nx * 2)); HC(hipMalloc(&b, nx * 2)); HC(hipMalloc(&cap, np));
        HC(hipMalloc(&old_cache, nc * 2)); HC(hipMalloc(&new_cache, nk)); HC(hipMalloc(&pg, page.size() * 4));
        HC(hipMemcpy(a, input.data(), nx * 2, hipMemcpyHostToDevice)); HC(hipMemcpy(b, input.data(), nx * 2, hipMemcpyHostToDevice));
        HC(hipMemcpy(cap, capture.data(), np, hipMemcpyHostToDevice)); HC(hipMemcpy(old_cache, cache.data(), nc * 2, hipMemcpyHostToDevice));
        HC(hipMemcpy(new_cache, packed.data(), nk, hipMemcpyHostToDevice)); HC(hipMemcpy(pg, page.data(), page.size() * 4, hipMemcpyHostToDevice));
        std::vector<RadParam> params = { pi("M", M), pi("n", n), pi("block", G), ps("format", "fp4_e2m1"),
            ps("scale", kind == 1 ? "e4m3" : "ue8m0"), ps("dtype", "bf16"), pi("block_size", bs), pi("kv_format", kind) };
        auto x = [&](void* p) { auto t = tensor((uint16_t*)p + guard, RAD_BF16, { M, n }); t.stride[0] = pitch; return t; };
        std::vector<RadTensor> old_rt = { x(a) }, new_rt = { x(b), tensor((uint8_t*)cap + guard, RAD_U8, { M, rb }) };
        int rc = call(plain_rt, old_rt, params); if (rc >= 0) rc = call(packed_rt, new_rt, params);
        std::vector<RadTensor> old_st = { x(a), tensor(pg, RAD_I32, { M }),
            tensor((uint16_t*)old_cache + guard, RAD_BF16, { blocks, 1, bs, n / bs }) };
        std::vector<RadTensor> new_st = { x(b), new_rt[1], tensor(pg, RAD_I32, { M }),
            tensor((uint8_t*)new_cache + guard, RAD_U8, { blocks, rb }) };
        if (rc >= 0) rc = call(plain_store, old_st, params);
        if (rc >= 0) rc = call(packed_store, new_st, params);
        HC(hipDeviceSynchronize());
        if (rc < 0) { std::fprintf(stderr, "FAIL producer/store rc %d kind%u M%d bs%d\n", rc, kind, M, bs); ++bad; }
        HC(hipMemcpy(ref.data(), a, nx * 2, hipMemcpyDeviceToHost)); HC(hipMemcpy(actual.data(), b, nx * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(got_cache.data(), old_cache, nc * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(got_capture.data(), cap, np, hipMemcpyDeviceToHost)); HC(hipMemcpy(got_packed.data(), new_cache, nk, hipMemcpyDeviceToHost));
        bad += ref != actual;
        for (size_t i = 0; i < nx; ++i) if (i < guard || i >= nx - guard || (i - guard) % pitch >= (size_t)n)
            bad += actual[i] != input[i];
        for (int i = 0; i < guard; ++i) {
            bad += got_capture[(size_t)i] != 0xA5 || got_capture[np - 1 - i] != 0xA5;
            bad += got_packed[(size_t)i] != 0xA5 || got_packed[nk - 1 - i] != 0xA5;
            bad += got_cache[(size_t)i] != 0x5AA5 || got_cache[nc - 1 - i] != 0x5AA5;
        }
        for (int p = 0; p < blocks; ++p) {
            if (!kc::decode_row(kind, got_packed.data() + guard + p * rb, n, decoded.data())) { ++bad; continue; }
            for (int c = 0; c < n; ++c) bad += decoded[(size_t)c] != got_cache[(size_t)(guard + p * n + c)];
        }
        for (void* p : { a, b, cap, old_cache, new_cache, pg }) HC(hipFree(p));
        ++cases;
    }
    dlclose(lib);
    std::printf("%s producer+packed-store: %zu uint16/guard differences\n", bad ? "FAIL" : "PASS", bad);
    if (bad == 0 && cases == 20)
        std::printf("PASS producer+packed-store coverage: cases=%zu kinds=1,2 blocks=1,2 M=1,7,33,257,512 exhaustive_bf16=65536\n", cases);
    return bad == 0 && cases == 20;
}
}
int main(int argc, char** argv) {
    if (argc > 2) { std::fprintf(stderr, "usage: %s [LIBR4D.so]\n", argv[0]); return 2; }
    std::vector<Desc> desc;
    for (int s = 1; s <= 126; ++s) desc.push_back({ 1, (uint8_t)s });
    for (int s = 1; s <= 253; ++s) desc.push_back({ 2, (uint8_t)s });
    desc.push_back({ 2, 255 }); desc.push_back({ 1, 0 }); desc.push_back({ 2, 0 });
    Desc* device = nullptr; Report* report = nullptr;
    HC(hipMalloc((void**)&device, desc.size() * sizeof(Desc))); HC(hipMalloc((void**)&report, sizeof(Report)));
    HC(hipMemcpy(device, desc.data(), desc.size() * sizeof(Desc), hipMemcpyHostToDevice)); HC(hipMemset(report, 0, sizeof(Report)));
    domain_oracle<<<(desc.size() + 127) / 128, 128>>>(device, (int)desc.size(), report);
    HC(hipDeviceSynchronize());
    Report got; HC(hipMemcpy(&got, report, sizeof(got), hipMemcpyDeviceToHost)); HC(hipFree(device)); HC(hipFree(report));
    const bool domain = got.cases == desc.size() && got.raw == 0 && got.bad == 0;
    std::printf("%s codec-domain: %u groups, normal%u pair%u half-ftz%u raw%u bad%u; actual minimum half bits +%04x -%04x\n",
                domain ? "PASS" : "FAIL", got.cases, got.normal, got.pair, got.ftz, got.raw, got.bad, got.half_pos, got.half_neg);
    const bool producer = argc == 2 ? producer_oracle(argv[1]) : true;
    return domain && producer ? 0 : 1;
}
