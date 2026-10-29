/* router_score_eval.cpp -- the read-ahead predictor's quantized scoring (RADIANCE_EXPERIMENTAL_DS41_
 * PREFETCH_SCORE, core/place/router_score.h) against its bf16 scoring, offline, on the container's
 * own routers.
 *
 * Reads every layer's router (f32 [384, 5120], bf16-valued), correction bias and ffn_norm gain from
 * a .rad container (abi/rad_format.h's layout, as scan_low_halves.py reads it; pread, the pages
 * dropped afterwards: ~320 MB, so run it outside a GPU window, under ionice -c3), builds the bf16,
 * int8 and int4 copies the predictor builds, and scores inputs as the predictor does:
 *
 *   recorded  RECORD (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RECORD, ab_window.sh's Pv part): the router
 *             inputs an arm's predictor was posted, each turned into its stand-in (x * g_{L+a} / g_L,
 *             a the file's lead) for layer L + a, as the predictor's worker does
 *   synthetic --synth N: N inputs a predicted layer, two kinds (no trace keeps real inputs yet):
 *             iid, z ~ N(0, 1) with three channels 30x (the residual stream's large ones), as
 *             x = bf16(z g_L) -- flat routing, harder than real; and aligned, the stand-in plus
 *             six of the predicted layer's router rows (each lifting its expert's logit by ~2-4
 *             sigma, as a real input selects its experts) -- decisive routing, likely easier
 *
 * For every input: bf16's own best k (the predictor without ..._PREFETCH_SCORE), and for int8 and
 * int4 at several rescore counts R the prediction the predictor makes (the quantized keys' best R
 * scored again in bf16, their best k): how often it is bf16's set, and in bf16's order, and where
 * bf16's picks ranked among the quantized keys. --time: one thread's time a prediction for each, the
 * layers taken in turn with 64 MB streamed between predictions (the engine's predictor finds its
 * tables cold: the pool streams experts between two layers); a quiet box gives an upper bound on
 * what the engine sees beside the pool's bursts.
 *
 * usage: router_score_eval MODEL.rad [RECORD] [--synth N] [--k K] [--max N] [--time N]
 * build: make router-eval (or g++ -O2 -std=c++17 -mavx2 -mfma -I../../core/place ...) */
#include "router_score.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rs = rad::rscore;

namespace {

constexpr int L = 40, E = 384, D = 5120;
constexpr int kRs[] = { 4, 6, 8, 12, 16, 24, 32 };
constexpr int kNR = (int)(sizeof kRs / sizeof kRs[0]);
constexpr int kEdges[] = { 4, 8, 12, 16, 24, 32 };
constexpr int kNB = (int)(sizeof kEdges / sizeof kEdges[0]) + 1;

[[noreturn]] void die(const char* f, const char* a = "") {
    std::fprintf(stderr, f, a);
    std::fputc('\n', stderr);
    std::exit(1);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* ---- the container (abi/rad_format.h, as scan_low_halves.py reads it) ---------------------------- */

struct Rad {
    int fd = -1;
    std::vector<uint8_t> strs, dir, planes, encs;
    uint64_t dir_count = 0;
    static constexpr int kEntry = 136, kEnc = 568;
    void pread_all(void* dst, uint64_t n, uint64_t off) const {
        uint8_t* p = static_cast<uint8_t*>(dst);
        while (n) {
            const ssize_t r = ::pread(fd, p, n, (off_t)off);
            if (r <= 0) die("short read in the container");
            p += r; n -= (uint64_t)r; off += (uint64_t)r;
        }
    }
    template <class T> static T at(const std::vector<uint8_t>& b, size_t off) {
        T v;
        std::memcpy(&v, b.data() + off, sizeof v);
        return v;
    }
    void open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) die("cannot open %s", path);
        uint8_t h[256];
        pread_all(h, sizeof h, 0);
        uint64_t f[10];
        std::memcpy(f, h + 56, sizeof f);
        strs.resize(f[1]); pread_all(strs.data(), f[1], f[0]);
        dir_count = f[5];
        dir.resize(f[5] * kEntry); pread_all(dir.data(), dir.size(), f[4]);
        planes.resize(f[7] * 16); pread_all(planes.data(), planes.size(), f[6]);
        encs.resize(f[9] * kEnc); pread_all(encs.data(), encs.size(), f[8]);
    }
    std::string name(size_t i) const {
        const uint64_t o = at<uint64_t>(dir, i * kEntry);
        if (!o || o >= strs.size()) return "";
        return std::string((const char*)strs.data() + o);
    }
    /* Entry `nm`'s one f32 plane into dst (n floats); false when absent or otherwise stored. */
    bool read_f32(const std::string& nm, float* dst, size_t n) const {
        for (size_t i = 0; i < dir_count; ++i) {
            if (name(i) != nm) continue;
            const size_t e = i * kEntry;
            const uint32_t enc = at<uint32_t>(dir, e + 24), n_planes = at<uint32_t>(dir, e + 92);
            const uint64_t first = at<uint64_t>(dir, e + 96);
            const uint32_t dtype = at<uint32_t>(encs, (size_t)enc * kEnc + 56 + 24);
            const uint64_t poff = at<uint64_t>(planes, first * 16), pbytes = at<uint64_t>(planes, first * 16 + 8);
            if (n_planes != 1 || dtype != 1 || pbytes != n * 4) return false;
            pread_all(dst, pbytes, poff);
            (void)::posix_fadvise(fd, (off_t)poff, (off_t)pbytes, POSIX_FADV_DONTNEED);
            return true;
        }
        return false;
    }
};

/* ---- the predictor's tables and scoring (router_predictor.cpp's rank()) ---------------------- */

struct Tables {
    std::vector<uint16_t> w;              /* [L][E][D] bf16 */
    std::vector<float> corr, gain;        /* [L][E], [L][D] */
    std::vector<int8_t> q8;               /* [L][E][D] */
    std::vector<uint8_t> q4;              /* [L][E][D / 2] */
    std::vector<float> s8, s4;            /* [L][E] */
};

using Key = std::pair<float, int32_t>;
bool best(const Key& a, const Key& b) { return a.first > b.first; }

float bf16_logit(const Tables& t, int l, int e, const float* xh) {
    return rs::dot_bf16_avx2(&t.w[((size_t)l * E + e) * D], xh, D);
}

/* bf16's own best k of layer l, and every expert's bf16 key. */
void bf16_rank(const Tables& t, int l, const float* xh, int k, int32_t* out, float* keys) {
    Key key[E];
    for (int e = 0; e < E; ++e) key[e] = { rs::key_of(bf16_logit(t, l, e, xh), t.corr[(size_t)l * E + e]), e };
    if (keys) for (int e = 0; e < E; ++e) keys[e] = key[e].first;
    std::partial_sort(key, key + k, key + E, best);
    for (int i = 0; i < k; ++i) out[i] = key[i].second;
}

/* Every expert's quantized key (bits 8 or 4). */
void quant_keys(const Tables& t, int bits, int l, const float* xh, float* qk) {
    alignas(32) int16_t xq[D];
    int64_t sum = 0;
    const float sx = rs::quantize_x16(xh, D, xq, &sum);
    for (int e = 0; e < E; ++e) {
        const size_t r = (size_t)l * E + e;
        const float c = t.corr[r];
        if (bits == 8) {
            const int64_t d = rs::dot_i8_avx2(&t.q8[r * D], xq, D);
            qk[e] = rs::key_of(t.s8[r] * sx * (float)d, c);
        } else {
            const int64_t d = rs::dot_u4_avx2(&t.q4[r * (D / 2)], xq, D);
            qk[e] = rs::key_of(t.s4[r] * sx * (float)(d - 8 * sum), c);
        }
    }
}

/* The predictor's pick from quantized keys: their best R keyed again in bf16 (bk: bf16's keys),
 * the best k of those. */
void rescored(const float* qk, const float* bk, int R, int k, int32_t* out) {
    Key key[E];
    for (int e = 0; e < E; ++e) key[e] = { qk[e], e };
    std::partial_sort(key, key + R, key + E, best);
    for (int i = 0; i < R; ++i) key[i].first = bk[key[i].second];
    std::partial_sort(key, key + k, key + R, best);
    for (int i = 0; i < k; ++i) out[i] = key[i].second;
}

struct Tally {
    uint64_t n = 0, set[kNR] = {}, order[kNR] = {}, rank[kNB] = {};
};

void tally(Tally* t, const int32_t* ref, const float* qk, const float* bk, int k) {
    ++t->n;
    for (int j = 0; j < kNR; ++j) {
        if (kRs[j] < k) continue;
        int32_t got[16];
        rescored(qk, bk, kRs[j], k, got);
        bool set = true, order = true;
        for (int i = 0; i < k; ++i) {
            order = order && got[i] == ref[i];
            set = set && std::find(got, got + k, ref[i]) != got + k;
        }
        t->set[j] += set;
        t->order[j] += order;
    }
    for (int i = 0; i < k; ++i) {
        int r = 0;
        for (int e = 0; e < E; ++e) r += qk[e] > qk[ref[i]];
        int b = 0;
        while (b < kNB - 1 && r >= kEdges[b]) ++b;
        ++t->rank[b];
    }
}

void print_tally(const char* what, const Tally& t, int k) {
    std::printf("%s: %llu predictions\n", what, (unsigned long long)t.n);
    std::printf("  rescored R      ");
    for (int j = 0; j < kNR; ++j) if (kRs[j] >= k) std::printf("%9d", kRs[j]);
    std::printf("\n  bf16's set      ");
    for (int j = 0; j < kNR; ++j) if (kRs[j] >= k) std::printf("%8.3f%%", 100.0 * t.set[j] / (double)std::max<uint64_t>(t.n, 1));
    std::printf("\n  bf16's order    ");
    for (int j = 0; j < kNR; ++j) if (kRs[j] >= k) std::printf("%8.3f%%", 100.0 * t.order[j] / (double)std::max<uint64_t>(t.n, 1));
    std::printf("\n  bf16's picks by their rank among the quantized keys:");
    for (int b = 0; b < kNB; ++b)
        b < kNB - 1 ? std::printf(" under %d %llu,", kEdges[b], (unsigned long long)t.rank[b])
                    : std::printf(" %d or more %llu\n", kEdges[b - 1], (unsigned long long)t.rank[b]);
}

}  /* namespace */

int main(int argc, char** argv) {
    if (argc < 2) die("usage: router_score_eval MODEL.rad [RECORD] [--synth N] [--k K] [--max N] [--time N]");
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) die("needs AVX2 and FMA");
    const char* model = argv[1];
    const char* record = nullptr;
    int synth = 0, k = 4, time_n = 0;
    long max_in = -1;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--synth" && i + 1 < argc) synth = std::atoi(argv[++i]);
        else if (a == "--k" && i + 1 < argc) k = std::clamp(std::atoi(argv[++i]), 1, 16);
        else if (a == "--max" && i + 1 < argc) max_in = std::atol(argv[++i]);
        else if (a == "--time" && i + 1 < argc) time_n = std::atoi(argv[++i]);
        else if (!record && a[0] != '-') record = argv[i];
        else die("unknown argument %s", argv[i]);
    }
    if (!record && !synth && !time_n) synth = 64;

    /* The routers, as the predictor's load reads them, and its copies. */
    Rad rad;
    rad.open(model);
    Tables t;
    t.w.resize((size_t)L * E * D);
    t.corr.resize((size_t)L * E);
    t.gain.resize((size_t)L * D);
    t.q8.resize((size_t)L * E * D);
    t.q4.resize((size_t)L * E * D / 2);
    t.s8.resize((size_t)L * E);
    t.s4.resize((size_t)L * E);
    std::vector<float> wf((size_t)E * D);
    double t_read = 0, t_q8 = 0, t_q4 = 0;
    for (int l = 0; l < L; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const double t0 = now_s();
        if (!rad.read_f32(b + "glm_router.weight", wf.data(), wf.size()) ||
            !rad.read_f32(b + "glm_router.correction", &t.corr[(size_t)l * E], E) ||
            !rad.read_f32(b + "ffn_norm.weight", &t.gain[(size_t)l * D], D))
            die("layer %s: its router, correction or ffn_norm is not one f32 plane of the predictor's shape",
                std::to_string(l).c_str());
        const double t1 = now_s();
        for (size_t i = 0; i < wf.size(); ++i) t.w[(size_t)l * E * D + i] = rs::f32_to_bf16_rne(wf[i]);
        for (int e = 0; e < E; ++e) {
            const size_t r = (size_t)l * E + e;
            t.s8[r] = rs::quantize_row_i8(&wf[(size_t)e * D], D, &t.q8[r * D]);
        }
        const double t2 = now_s();
        for (int e = 0; e < E; ++e) {
            const size_t r = (size_t)l * E + e;
            t.s4[r] = rs::quantize_row_u4(&wf[(size_t)e * D], D, &t.q4[r * (D / 2)]);
        }
        const double t3 = now_s();
        t_read += t1 - t0; t_q8 += t2 - t1; t_q4 += t3 - t2;
    }
    std::printf("routers: %d layers from %s (%.0f MB read in %.1f s); quantized: int8 %.0f MB in %.2f s, "
                "int4 %.0f MB in %.2f s\n", L, model, (double)L * E * D * 4 / 1e6, t_read,
                (double)t.q8.size() / 1e6, t_q8, (double)t.q4.size() / 1e6, t_q4);

    /* The stand-in for layer l + a from layer l's input x: bf16(x) * g_{l+a} / g_l, in the
     * predictor's float operations. */
    auto standin = [&](const uint16_t* x, int l, int a, float* xh) {
        for (int c = 0; c < D; ++c) {
            const float g0 = t.gain[(size_t)l * D + c], ga = t.gain[(size_t)(l + a) * D + c];
            xh[c] = rs::bf16_to_f32(x[c]) * (g0 != 0.0f ? ga / g0 : 0.0f);
        }
    };
    std::vector<float> xh(D), bk(E), qk(E);
    int32_t ref[16];
    auto score_one = [&](int target, Tally* t8, Tally* t4) {
        bf16_rank(t, target, xh.data(), k, ref, bk.data());
        quant_keys(t, 8, target, xh.data(), qk.data());
        tally(t8, ref, qk.data(), bk.data(), k);
        quant_keys(t, 4, target, xh.data(), qk.data());
        tally(t4, ref, qk.data(), bk.data(), k);
    };

    if (record) {
        const int fd = ::open(record, O_RDONLY);
        if (fd < 0) die("cannot open %s", record);
        uint32_t h[16];
        if (::pread(fd, h, sizeof h, 0) != (ssize_t)sizeof h || std::memcmp(h, "RADPRDR1", 8) || h[2] != 1 ||
            h[3] != (uint32_t)L || h[4] != (uint32_t)E || h[5] != (uint32_t)D)
            die("%s: not a RADPRDR1 record of 40 x 384 x 5120 routers", record);
        const int ahead = (int)h[6];
        Tally t8, t4;
        std::vector<uint8_t> rec(8 + 2 * D);
        std::map<int, Tally> by_layer8;
        off_t off = sizeof h;
        for (long n = 0; max_in < 0 || n < max_in; ++n, off += (off_t)rec.size()) {
            if (::pread(fd, rec.data(), rec.size(), off) != (ssize_t)rec.size()) break;
            int32_t layer;
            std::memcpy(&layer, rec.data() + 4, 4);
            if (layer < 0 || layer + ahead >= L) continue;
            standin(reinterpret_cast<const uint16_t*>(rec.data() + 8), layer, ahead, xh.data());
            score_one(layer + ahead, &t8, &t4);
        }
        ::close(fd);
        std::printf("recorded inputs (%s, layer L predicts L+%d), top %d:\n", record, ahead, k);
        print_tally("int8", t8, k);
        print_tally("int4", t4, k);
    }
    if (synth > 0) {
        /* z ~ N(0, 1), three channels 30x, as layer l's input bf16(z g_l), for every layer l + 2;
         * aligned: the same stand-in plus six of layer l + 2's router rows, each scaled to lift its
         * expert's logit by 2-4 standard deviations of the others'. */
        uint64_t s = 0x9E3779B97F4A7C15ull;
        auto uni = [&] { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return (float)((s * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f; };
        auto gauss = [&] { return (uni() + uni() + uni() + uni() - 2.0f) * 1.7320508f; };
        Tally t8, t4, a8, a4;
        std::vector<uint16_t> x(D);
        std::vector<float> lg(E);
        for (int target = 2; target < L; ++target)
            for (int i = 0; i < synth; ++i) {
                const int l = target - 2;
                for (int c = 0; c < D; ++c) {
                    float z = gauss();
                    if (c == 101 || c == 2047 || c == 4093) z *= 30.0f;
                    x[(size_t)c] = rs::f32_to_bf16_rne(z * t.gain[(size_t)l * D + c]);
                }
                standin(x.data(), l, 2, xh.data());
                score_one(target, &t8, &t4);
                /* the logits' spread on this input, then six experts lifted along their own rows */
                double m = 0, v = 0;
                for (int e = 0; e < E; ++e) { lg[(size_t)e] = bf16_logit(t, target, e, xh.data()); m += lg[(size_t)e]; }
                m /= E;
                for (int e = 0; e < E; ++e) v += (lg[(size_t)e] - m) * (lg[(size_t)e] - m);
                const double sd = std::sqrt(v / E);
                for (int j = 0; j < 6; ++j) {
                    const int e = (int)(uni() * E) % E;
                    const uint16_t* w = &t.w[((size_t)target * E + e) * D];
                    double n2 = 0;
                    for (int c = 0; c < D; ++c) { const double a = rs::bf16_to_f32(w[c]); n2 += a * a; }
                    const float lift = (float)((2.0 + 2.0 * uni()) * sd / std::max(n2, 1e-30));
                    for (int c = 0; c < D; ++c) xh[(size_t)c] += lift * rs::bf16_to_f32(w[c]);
                }
                score_one(target, &a8, &a4);
            }
        std::printf("synthetic inputs, iid (%d a layer, layer L predicts L+2), top %d:\n", synth, k);
        print_tally("int8", t8, k);
        print_tally("int4", t4, k);
        std::printf("synthetic inputs, aligned (six experts lifted 2-4 sigma), top %d:\n", k);
        print_tally("int8", a8, k);
        print_tally("int4", a4, k);
    }
    if (time_n > 0) {
        /* One thread, the layers in turn, 64 MB streamed between predictions. */
        std::vector<uint8_t> flush((size_t)64 << 20, 1);
        volatile uint64_t sink = 0;
        auto stream = [&] { uint64_t a = 0; for (size_t i = 0; i < flush.size(); i += 64) a += flush[i]; sink = sink + a; };
        std::vector<float> xs(D);
        for (int c = 0; c < D; ++c) xs[(size_t)c] = std::sin(0.37f * (float)c);
        double tb = 0, t8 = 0, t4 = 0, r8 = 0, r4 = 0;
        int32_t out[16];
        for (int i = 0; i < time_n; ++i) {
            const int target = 2 + i % (L - 2);
            stream();
            double a = now_s();
            bf16_rank(t, target, xs.data(), k, out, nullptr);
            tb += now_s() - a;
            for (int bits : { 8, 4 }) {
                stream();
                a = now_s();
                quant_keys(t, bits, target, xs.data(), qk.data());
                Key key[E];
                for (int e = 0; e < E; ++e) key[e] = { qk[(size_t)e], e };
                std::partial_sort(key, key + 12, key + E, best);
                const double b = now_s();
                for (int j = 0; j < 12; ++j)
                    key[j].first = rs::key_of(bf16_logit(t, target, key[j].second, xs.data()),
                                              t.corr[(size_t)target * E + key[j].second]);
                std::partial_sort(key, key + k, key + 12, best);
                const double c = now_s();
                (bits == 8 ? t8 : t4) += b - a;
                (bits == 8 ? r8 : r4) += c - b;
            }
        }
        std::printf("one thread, tables cold (64 MB streamed between predictions), %d predictions: bf16 "
                    "%.1f us a prediction; int8 %.1f us + %.1f us rescoring 12; int4 %.1f us + %.1f us\n",
                    time_n, tb * 1e6 / time_n, t8 * 1e6 / time_n, r8 * 1e6 / time_n, t4 * 1e6 / time_n,
                    r4 * 1e6 / time_n);
        (void)sink;
    }
    ::close(rad.fd);
    return 0;
}
