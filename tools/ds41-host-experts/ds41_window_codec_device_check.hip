/* SOURCE oracle; execute ONLY in a two-lock stopped-QFN GPU window.
 * Usage: ds41_window_codec_device_check LIBR4D.so
 * Requires the ACTUAL old quant producer, not just a duplicated arithmetic test.
 * Strict BF16 uint16 comparison, including NaNs and signed zeros. ZeroRaw is
 * mandatory. This does NOT gate future fused capture/store/readers or pages.
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rad_plugin.h"
#include "../../arch/common/rad_ds41_window_codec.h"
namespace wc = rad_ds41_window_codec;
#define HC(x) do { const hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr,"FAIL %s: %s\n",#x,hipGetErrorString(e)); std::exit(2); } } while(0)
namespace {
__device__ float old_e4(uint8_t b) {
    const uint32_t s = uint32_t(b >> 7) << 31, e = (b >> 3) & 15;
    uint32_t m = b & 7, u;
    if (!e) {
        if (!m) u = s;
        else { int sh = 0; while (!(m & 8)) { m <<= 1; ++sh; }
            u = s | (uint32_t(127 - 6 - sh) << 23) | ((m & 7) << 20); }
    } else if (e == 15 && m == 7) u = s | 0x7FC00000u;
    else u = s | ((e + 120) << 23) | (m << 20);
    return __uint_as_float(u);
}
__device__ uint8_t old_code(float f) {
    const uint32_t u = __float_as_uint(f), s = (u >> 24) & 128, a = u & 0x7FFFFFFF;
    if (a > 0x7F800000) return uint8_t(s | 127);
    if (a >= 0x43E00000) return uint8_t(s | 126);
    const int e = int(a >> 23) - 127; uint32_t m = a & 0x7FFFFF;
    if (e < -10) return uint8_t(s);
    if (e < -6) {
        const int sh = -6 - e + 20; m |= 0x800000;
        const uint32_t lo = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        uint32_t q = m >> sh; if (lo > half || (lo == half && (q & 1))) ++q;
        return uint8_t(s | q);
    }
    uint32_t q = m >> 20; const uint32_t lo = m & 0xFFFFF; int ee = e + 7;
    if (lo > 0x80000 || (lo == 0x80000 && (q & 1))) ++q;
    if (q == 8) { q = 0; ++ee; }
    return uint8_t(s | (ee > 15 || (ee == 15 && q >= 7) ? 126 : (uint32_t(ee) << 3) | q));
}
__device__ uint16_t old_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7FFFFFFF) > 0x7F800000) return uint16_t((u >> 16) | 0x40);
    return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
struct Report { unsigned groups = 0, bad = 0, raw = 0, pair = 0, scale_bad = 0; };
__global__ void domain(Report* report) {
    const int idx = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= 143 * 8 + 2) return;
    uint8_t scale = idx < 143 * 8 ? uint8_t(105 + idx / 8) : uint8_t(255);
    uint8_t codes[32], packed[32], stored = 0xA5;
    uint16_t actual[32];
    for (int i = 0; i < 32; ++i) {
        // Inf source can produce signed zero or signed canonical qNaN only.
        codes[i] = scale == 255 ? uint8_t((i & 1 ? 128 : 0) | (i & 2 ? 127 : 0))
                               : uint8_t((idx % 8) * 32 + i);
        actual[i] = old_bf16(old_e4(codes[i]) * __uint_as_float(uint32_t(scale) << 23));
    }
    const auto result = wc::pack_captured_group(scale, codes, actual, packed, &stored);
    atomicAdd(&report->groups, 1u);
    if (result != wc::PackResult::Exact) { atomicAdd(&report->raw, 1u); return; }
    if (!stored) atomicAdd(&report->pair, 1u);
    for (int i = 0; i < 32; ++i) { uint16_t got = 0;
        if (!wc::decode_group_bits(packed, stored, i, &got) || got != actual[i]) atomicAdd(&report->bad, 1u); }
}

/* Independent transcription of ORIGINAL producer arithmetic over PRE-quant BF16.
 * Captures codes/scales while retaining a separately compared expected output.
 * Serial positive amax equals the current XOR max; NaNs contribute zero. */
__global__ void capture_source(const uint16_t* input, long pitch, uint8_t* captured,
                               uint16_t* expected, int M, int n, int packed_pitch) {
    const int g = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= M * (n / 32)) return;
    const int r = g / (n / 32), group = g % (n / 32), start = group * 32;
    float amax = 0;
    for (int i = 0; i < 32; ++i) {
        float v = fabsf(__uint_as_float(uint32_t(input[r * pitch + start + i]) << 16));
        v = v == v ? v : 0.0f; amax = amax < v ? v : amax;
    }
    amax = amax < 1e-4f ? 1e-4f : amax;
    const float inv = 1.0f / 448.0f;
    const uint32_t a = __float_as_uint(amax * inv);
    const int exponent = int((a >> 23) & 255) - 127 + ((a & 0x7FFFFF) ? 1 : 0);
    const float scale = __uint_as_float(uint32_t(exponent + 127) << 23);
    captured[r * packed_pitch + n + group] = uint8_t(__float_as_uint(scale) >> 23);
    for (int i = 0; i < 32; ++i) {
        const float v = __uint_as_float(uint32_t(input[r * pitch + start + i]) << 16) / scale;
        const float clamp = v < -448.0f ? -448.0f : 448.0f < v ? 448.0f : v;
        const uint8_t code = old_code(clamp);
        captured[r * packed_pitch + start + i] = code;
        expected[r * pitch + start + i] = old_bf16(old_e4(code) * scale);
    }
}
__global__ void finalize(const uint8_t* captured, const uint16_t* actual, long pitch,
                         uint8_t* output, int M, int n, int pp, Report* report) {
    const int g = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (g >= M * (n / 32)) return;
    const int r = g / (n / 32), group = g % (n / 32), start = group * 32;
    const uint8_t scale = captured[r * pp + n + group];
    atomicAdd(&report->groups, 1u);
    if ((!wc::normal_scale(scale) && scale != 255) || !scale) atomicAdd(&report->scale_bad, 1u);
    const auto result = wc::pack_captured_group(scale, captured + r * pp + start,
        actual + r * pitch + start, output + r * pp + start, output + r * pp + n + group);
    if (result != wc::PackResult::Exact) { atomicAdd(&report->raw, 1u); return; }
    if (!output[r * pp + n + group]) atomicAdd(&report->pair, 1u);
    for (int i = 0; i < 32; ++i)
        if (wc::decode_row_bits(output + r * pp, n, start + i) != actual[r * pitch + start + i])
            atomicAdd(&report->bad, 1u);
}
using Launch = int (*)(const RadArgs*, RadStream);
RadParam pi(const char* name, int64_t v) { RadParam p{}; p.key = name; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam ps(const char* name, const char* v) { RadParam p{}; p.key = name; p.kind = RAD_P_STR; p.sval = v; return p; }
}
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr,"usage: %s LIBR4D.so\n",argv[0]); return 2; }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr,"FAIL dlopen: %s\n",dlerror()); return 2; }
    const auto quant = (Launch)dlsym(lib,"r4d_rad_quant_roundtrip");
    if (!quant) { std::fprintf(stderr,"FAIL missing original quant_roundtrip\n"); return 2; }
    Report* report = nullptr; HC(hipMalloc((void**)&report,sizeof(Report))); HC(hipMemset(report,0,sizeof(Report)));
    domain<<<(143 * 8 + 2 + 127) / 128,128>>>(report);
    HC(hipDeviceSynchronize()); Report got{}; HC(hipMemcpy(&got,report,sizeof got,hipMemcpyDeviceToHost));
    bool okay = got.groups == 143 * 8 + 2 && !got.bad && !got.raw && !got.scale_bad;
    std::printf("%s window-fp8-domain groups=%u raw=%u bad=%u pair=%u\n",okay?"PASS":"FAIL",got.groups,got.raw,got.bad,got.pair);
    constexpr int M=4096,n=512,pitch=528,pp=wc::row_bytes(n)+16,guard=32;
    std::vector<uint16_t> input(size_t(M)*pitch+2*guard,0x5AA5), old(input.size()), expected(input.size());
    uint16_t *pre=nullptr,*actual=nullptr,*expect=nullptr; uint8_t *capture=nullptr,*packed=nullptr;
    for (auto p : {&pre,&actual,&expect}) HC(hipMalloc((void**)p,input.size()*2));
    const size_t np=size_t(M)*pp+2*guard;
    HC(hipMalloc((void**)&capture,np)); HC(hipMalloc((void**)&packed,np));
    const std::array<uint16_t,12> edge={0,0x8000,1,0x8001,0x007F,0x807F,0x7F7F,0xFF7F,0x7F80,0xFF80,0x7FC1,0xFFFF};
    for (int profile=0;profile<4;++profile) {
        for(int r=0;r<M;++r)for(int g=0;g<16;++g)for(int i=0;i<32;++i) {
            const uint16_t v=uint16_t(r*16+g); uint16_t x=v;
            if(i && profile==1)x=edge[size_t(i)%edge.size()];
            if(i && profile==2)x=uint16_t(v ^ uint16_t(i*0x631));
            if(i && profile==3)x=edge[size_t((i+g)%6+6)];
            input[guard+size_t(r)*pitch+g*32+i]=x;
        }
        for(auto p:{pre,actual,expect})HC(hipMemcpy(p,input.data(),input.size()*2,hipMemcpyHostToDevice));
        HC(hipMemset(capture,0xA5,np));HC(hipMemset(packed,0xA5,np));HC(hipMemset(report,0,sizeof(Report)));
        capture_source<<<(M*16+127)/128,128>>>(pre+guard,pitch,capture+guard,expect+guard,M,n,pp);
        RadTensor t{};t.data=actual+guard;t.dtype=RAD_BF16;t.rank=2;t.shape[0]=M;t.shape[1]=n;t.stride[0]=pitch;t.stride[1]=1;
        std::array<RadParam,6> p={pi("M",M),pi("n",n),pi("block",32),ps("format","fp8_e4m3"),ps("scale","ue8m0"),ps("dtype","bf16")};
        RadArgs a{};a.t=&t;a.n_t=1;a.p=p.data();a.n_p=int(p.size());
        const int rc=quant(&a,nullptr);
        if(rc<0){std::fprintf(stderr,"FAIL original producer rc=%d\n",rc);okay=false;break;}
        finalize<<<(M*16+127)/128,128>>>(capture+guard,actual+guard,pitch,packed+guard,M,n,pp,report);
        HC(hipDeviceSynchronize());HC(hipMemcpy(&got,report,sizeof got,hipMemcpyDeviceToHost));
        HC(hipMemcpy(old.data(),actual,input.size()*2,hipMemcpyDeviceToHost));HC(hipMemcpy(expected.data(),expect,input.size()*2,hipMemcpyDeviceToHost));
        std::vector<uint8_t> cap(np),out(np);HC(hipMemcpy(cap.data(),capture,np,hipMemcpyDeviceToHost));HC(hipMemcpy(out.data(),packed,np,hipMemcpyDeviceToHost));
        size_t bad=old!=expected;
        for(size_t i=0;i<input.size();++i)if(i<guard||i>=input.size()-guard||(i-guard)%pitch>=n)bad+=old[i]!=input[i]||expected[i]!=input[i];
        for(size_t i=0;i<np;++i)if(i<guard||i>=np-guard||(i-guard)%pp>=size_t(wc::row_bytes(n)))bad+=cap[i]!=0xA5||out[i]!=0xA5;
        std::vector<uint16_t> decoded(n);
        for(int r=0;r<M;++r){if(!wc::decode_row(out.data()+guard+size_t(r)*pp,n,decoded.data())){++bad;continue;}
            for(int c=0;c<n;++c)bad+=decoded[c]!=old[guard+size_t(r)*pitch+c];}
        const bool pass=got.groups==M*16&&!got.raw&&!got.bad&&!got.scale_bad&&!bad;
        okay&=pass;
        std::printf("%s window-fp8-original-producer profile=%d groups=%u raw=%u bad=%u scale_bad=%u byte_guard_bad=%zu pair=%u\n",
            pass?"PASS":"FAIL",profile,got.groups,got.raw,got.bad,got.scale_bad,bad,got.pair);
    }
    for(void* p:{(void*)pre,(void*)actual,(void*)expect,(void*)capture,(void*)packed,(void*)report})HC(hipFree(p));
    dlclose(lib);std::printf("%s ds41_window_codec_exact\n",okay?"PASS":"FAIL");return okay?0:1;
}
