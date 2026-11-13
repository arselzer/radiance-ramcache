/* tests/r4d_emu: the QSA work list (qsa_work, r4d_qsa_work_bf16.hip) on the host SIMT emulator against
 * a host walk of the batch, at sequence counts past the launcher's old bound of 256: a decode step of
 * 512 or 1024 sequences is what a batch job with --max-num-seqs 512 and --max-seq-len steps (window
 * ab-20261113T054538Z: "qsa_work_f32 ... refused op 'qsa_work' at M=512 ... seqs=512"). Every output
 * the block-key kernel reads is held to the walk: each completing block's page and first-token
 * position, its `ratio` staged key rows (from this step or the sequence's tail slot), the complete-
 * block count a sequence, and page -1 in the slots past the real count. f32 planes (the compressor's)
 * and bf16. Mixed steps put multi-token chunks among the decode rows, so a block straddles tail and
 * chunk. The walk is the kernel's contract, so the same rows below 256 sequences pin that nothing
 * changed there. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    return t;
}

/* One step: `seqs` sequences, sequence s with len[s] tokens from position start[s]. Returns failures. */
int run(const std::string& name, bool f32, int64_t seqs, int ratio, int ring, std::vector<int> len,
        std::vector<int> start) {
    const int64_t n = 16, esz = f32 ? 4 : 2, slots = seqs + 5, btw = 160;
    std::vector<int32_t> cu((size_t)seqs + 1, 0);
    for (int64_t s = 0; s < seqs; ++s) cu[(size_t)s + 1] = cu[(size_t)s] + len[(size_t)s];
    const int64_t T = cu[(size_t)seqs];
    std::vector<int32_t> pos((size_t)T), sidx((size_t)seqs), bt((size_t)(seqs * btw));
    for (int64_t s = 0; s < seqs; ++s) {
        for (int i = 0; i < len[(size_t)s]; ++i) pos[(size_t)(cu[(size_t)s] + i)] = start[(size_t)s] + i;
        sidx[(size_t)s] = (int32_t)((s * 7 + 3) % slots);
        for (int64_t b = 0; b < btw; ++b) bt[(size_t)(s * btw + b)] = (int32_t)((s * 211 + b * 13 + 5) % 100003);
    }
    /* Element bits are moved, never computed: distinct 16- or 32-bit patterns a key. */
    std::vector<uint8_t> k((size_t)(T * n * esz)), tail((size_t)(slots * ring * n * esz));
    for (size_t i = 0; i < k.size(); ++i) k[i] = (uint8_t)(i * 131 + 7);
    for (size_t i = 0; i < tail.size(); ++i) tail[i] = (uint8_t)(i * 89 + 201);
    struct Wk { int64_t s, b; };
    std::vector<Wk> want;
    for (int64_t s = 0; s < seqs; ++s) {
        if (len[(size_t)s] <= 0) continue;
        const int64_t p0 = pos[(size_t)cu[(size_t)s]], p1 = pos[(size_t)cu[(size_t)s + 1] - 1];
        for (int64_t b = p0 / ratio; b <= (p1 + 1) / ratio - 1; ++b) want.push_back({ s, b });
    }
    const int64_t W = (T + ratio - 1) / ratio + seqs;
    std::vector<uint8_t> stage((size_t)(W * ratio * n * esz), 0xEE);
    std::vector<int32_t> page((size_t)W, 0x55555555), bpos((size_t)W, 0x55555555), nc((size_t)seqs, -7);
    const uint32_t dt = f32 ? RAD_F32 : RAD_BF16;
    RadTensor t[11] = {
        tens(k.data(), dt, { T, n }), tens(pos.data(), RAD_I32, { T }), tens(cu.data(), RAD_I32, { seqs + 1 }),
        tens(tail.data(), dt, { slots, 1, ring, n }), tens(sidx.data(), RAD_I32, { seqs, 1 }),
        tens(bt.data(), RAD_I32, { seqs, btw }), tens(nullptr, RAD_I32, { 3, T }),
        tens(stage.data(), dt, { W * ratio, n }), tens(page.data(), RAD_I32, { W }),
        tens(bpos.data(), RAD_I32, { W }), tens(nc.data(), RAD_I32, { seqs }),
    };
    RadParam p[7]{};
    p[0] = { "M", RAD_P_INT, T }; p[1] = { "n", RAD_P_INT, n }; p[2] = { "ratio", RAD_P_INT, ratio };
    p[3] = { "ring", RAD_P_INT, ring }; p[4] = { "seqs", RAD_P_INT, seqs }; p[5] = { "work", RAD_P_INT, W };
    p[6] = { "dtype", RAD_P_STR, 0 }; p[6].sval = f32 ? "f32" : "bf16";
    RadArgs a{}; a.t = t; a.n_t = 11; a.p = p; a.n_p = 7;
    const int rc = r4d_rad_qsa_work(&a, nullptr);
    if (rc != RAD_OK) { std::printf("FAIL %s: launch rc=%d\n", name.c_str(), rc); return 1; }
    long bad = 0;
    for (int64_t s = 0; s < seqs; ++s) {
        const int32_t w2 = len[(size_t)s] > 0 ? (pos[(size_t)cu[(size_t)s + 1] - 1] + 1) / ratio : 0;
        bad += nc[(size_t)s] != w2;
    }
    for (int64_t w = 0; w < W; ++w) {
        if (w >= (int64_t)want.size()) { bad += page[(size_t)w] != -1; continue; }
        const int64_t s = want[(size_t)w].s, b = want[(size_t)w].b, p0 = pos[(size_t)cu[(size_t)s]];
        bad += page[(size_t)w] != bt[(size_t)(s * btw + b)];
        bad += bpos[(size_t)w] != (int32_t)(b * ratio);
        for (int64_t i = 0; i < ratio; ++i) {
            const int64_t q = b * ratio + i;
            const uint8_t* src = q >= p0 ? &k[(size_t)((cu[(size_t)s] + (q - p0)) * n * esz)]
                                         : &tail[(size_t)(((sidx[(size_t)s] * ring) + q % ring) * n * esz)];
            bad += std::memcmp(&stage[(size_t)((w * ratio + i) * n * esz)], src, (size_t)(n * esz)) != 0;
        }
    }
    std::printf("%s %s: %lld sequences, %lld tokens, %zu blocks, %ld mismatch(es)\n", bad ? "FAIL" : "ok  ",
                name.c_str(), (long long)seqs, (long long)T, want.size(), bad);
    return bad ? 1 : 0;
}

/* A decode step: one token a sequence, positions spread so about one in `ratio` completes a block. */
int decode(bool f32, int64_t seqs, int ratio, int ring) {
    std::vector<int> len((size_t)seqs, 1), start((size_t)seqs);
    for (int64_t s = 0; s < seqs; ++s) start[(size_t)s] = (int)((s * 37 + 11) % 600);
    return run(std::string(f32 ? "f32" : "bf16") + " decode", f32, seqs, ratio, ring, len, start);
}

/* Decode rows with chunks among them (and an idle row of zero tokens), seeded. */
int mixed(bool f32, int64_t seqs, int ratio, int ring, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<int> len((size_t)seqs), start((size_t)seqs);
    for (int64_t s = 0; s < seqs; ++s) {
        const unsigned r = g() % 16;
        len[(size_t)s] = r == 0 ? 0 : r < 13 ? 1 : 2 + (int)(g() % 9);
        start[(size_t)s] = (int)(g() % 700);
    }
    return run(std::string(f32 ? "f32" : "bf16") + " mixed seed " + std::to_string(seed), f32, seqs, ratio,
               ring, len, start);
}

}  // namespace

int main() {
    int fails = 0;
    for (int64_t seqs : { 4, 256, 257, 512, 1024 }) {
        fails += decode(true, seqs, 2, 2);    /* DeepSeek-V4.1's compressor: ratio 2, ring 2 */
        fails += decode(false, seqs, 4, 7);   /* a ring past the ratio */
    }
    for (unsigned seed = 1; seed <= 3; ++seed) {
        fails += mixed(true, 512, 2, 2, seed);
        fails += mixed(false, 1024, 4, 6, seed);
    }
    std::printf("qsa_work: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
