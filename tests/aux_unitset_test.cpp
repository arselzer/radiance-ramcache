/* aux_unitset_test.cpp -- the aux expert worker's unit set (tools/ds41-host-experts/ds41_aux.h; work/aux/SPEC.md
 * §5), pure, on four fabricated rank plans:
 *  - exclusion (candidate()): lent, slab, host-planes-ineligible, draft (not trunk), refused, an unusable layer,
 *    wrong weight sizes or counts;
 *  - the order: profile share desc, expert asc, layer asc, rank asc;
 *  - the capacity cut by cap and stride (layout() at the packed cache's default slots), and the ranks' quotas;
 *  - determinism: any input order gives the same set. */
#include "rad_test.h"
#include "place/aux_expert_worker.h"

#include <algorithm>
#include <random>
#include <vector>

using namespace ds41aux;

namespace {

UnitFacts ok_unit() {
    UnitFacts f;
    f.layer_ok = f.trunk = f.planes_eligible = true;
    f.n_weights = 2;
    f.bytes[0] = kGuBlockBytes;
    f.bytes[1] = kDnBlockBytes;
    f.expert = 7;
    return f;
}

/* A rank's plan: 40 routed layers x 96 experts, a fabricated share per unit (rank-invariant inputs: the share
 * depends on layer and expert alone, as the sealed profile's does, so ties across ranks are by rank). */
std::vector<Cand> plan(int rank) {
    std::vector<Cand> v;
    for (int layer = 0; layer < 40; ++layer)
        for (int e = 0; e < 96; ++e) {
            Cand c;
            c.rank = rank;
            c.unit = layer * 96 + e;
            c.layer = layer;
            c.expert = e;
            c.share = (float)((e * 37 + layer * 11 + rank * 5) % 97) / 1000.0f;
            v.push_back(c);
        }
    return v;
}

}  // namespace

TEST(exclusion) {
    CHECK(candidate(ok_unit()));
    UnitFacts f = ok_unit();
    std::swap(f.bytes[0], f.bytes[1]);   /* down first: the order of weights does not matter */
    CHECK(candidate(f));
    f = ok_unit(); f.lent = true;            CHECK(!candidate(f));
    f = ok_unit(); f.in_slab = true;         CHECK(!candidate(f));
    f = ok_unit(); f.planes_eligible = false; CHECK(!candidate(f));
    f = ok_unit(); f.trunk = false;          CHECK(!candidate(f));   /* a draft head's layer */
    f = ok_unit(); f.refused = true;         CHECK(!candidate(f));
    f = ok_unit(); f.layer_ok = false;       CHECK(!candidate(f));
    f = ok_unit(); f.n_weights = 1;          CHECK(!candidate(f));
    f = ok_unit(); f.bytes[1] = kGuBlockBytes; CHECK(!candidate(f));
    f = ok_unit(); f.expert = 96;            CHECK(!candidate(f));
    f = ok_unit(); f.expert = -1;            CHECK(!candidate(f));
}

TEST(the_order_share_expert_layer_rank) {
    Cand a, b;
    a.share = 0.5f; b.share = 0.4f;
    CHECK(before(a, b));
    CHECK(!before(b, a));
    b.share = 0.5f;
    a.expert = 3; b.expert = 4;
    CHECK(before(a, b));
    b.expert = 3;
    a.layer = 1; b.layer = 2;
    CHECK(before(a, b));
    b.layer = 1;
    a.rank = 0; b.rank = 1;
    CHECK(before(a, b));
    CHECK(!before(b, a));
}

TEST(layout_and_capacity) {
    const Layout L = layout(11999296, 6001984);
    CHECK_EQ(L.dn_off, (int64_t)11999488);
    CHECK_EQ(L.stride, (int64_t)18001920);
    CHECK_EQ(L.dn_off % 256, (int64_t)0);
    CHECK_EQ(L.stride % 4096, (int64_t)0);
    const Ws w = ws();
    CHECK_EQ(w.rec % 4096, (int64_t)0);
    CHECK_EQ(w.bytes % 4096, (int64_t)0);
    CHECK(w.x >= 1024);
    CHECK(w.rec >= w.x + (int64_t)kMaxRows * kEmbd * 2);
    CHECK_EQ(w.out, (int64_t)kMaxSlots * kEmbd * 2);
    CHECK(w.bytes < 3 * ((int64_t)1 << 20));   /* about 2.4 MiB a lane */
    const int64_t fixed = fixed_bytes(4, 40);
    CHECK_EQ(fixed, tables_bytes(4, 40) + 4 * w.bytes);
    CHECK_EQ(tables_bytes(4, 40), align_up((int64_t)4 * 40 * 2 * 96 * 8, 4096));
    /* 26000 MiB at the default slots: about 1,513 units. */
    const int64_t n = capacity(26000, 4, 40, L.stride, 1000000);
    CHECK_EQ(n, (((int64_t)26000 << 20) - fixed) / L.stride);
    CHECK(n >= 1500 && n <= 1520);
    CHECK_EQ(capacity(26000, 4, 40, L.stride, 100), (int64_t)100);   /* fewer candidates than room */
    CHECK_EQ(capacity(1, 4, 40, L.stride, 100), (int64_t)0);         /* the fixed part does not fit */
}

/* RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS (ds41_aux.h, "taller passes"): ws(16) is ws() byte for byte (today's record:
 * sorted' at 0, off' at 512, x at 1024); ws(rows) holds rows token rows and rows * 6 slots in every region, each
 * 4 KiB aligned and in order; the fixed part and the capacity follow the lanes' size; no workspace past the cap. */
TEST(taller_workspaces) {
    const Ws a = ws(), b = ws(16);
    CHECK_EQ(a.sorted, (int64_t)0);
    CHECK_EQ(a.off, (int64_t)512);
    CHECK_EQ(a.x, (int64_t)1024);
    CHECK(a.sorted == b.sorted && a.off == b.off && a.x == b.x && a.rec == b.rec && a.gu_out == b.gu_out &&
          a.act == b.act && a.dn_out == b.dn_out && a.bytes == b.bytes && a.out == b.out);
    CHECK_EQ(fixed_bytes(4, 40), fixed_bytes(4, 40, 16));
    const Layout L = layout(11999296, 6001984);
    CHECK_EQ(capacity(26000, 4, 40, L.stride, 1000000), capacity(26000, 4, 40, L.stride, 1000000, 16));
    int64_t prev = a.bytes;
    for (int rows : { 17, 32, 64, 128, 256, 512 }) {
        const Ws w = ws(rows);
        const int64_t slots = (int64_t)rows * kTopK;
        CHECK_EQ(w.sorted, (int64_t)0);
        CHECK(w.off >= slots * 4 && w.off % 512 == 0);
        CHECK(w.x >= w.off + (kExperts + 1) * 4 && w.x % 512 == 0);
        CHECK(w.rec >= w.x + (int64_t)rows * kEmbd * 2 && w.rec % 4096 == 0);
        CHECK_EQ(w.gu_out, w.rec);
        CHECK(w.act >= w.gu_out + slots * kGuN * 2 && w.act % 4096 == 0);
        CHECK(w.dn_out >= w.act + slots * kFf * 2 && w.dn_out % 4096 == 0);
        CHECK(w.bytes >= w.dn_out + slots * kEmbd * 2 && w.bytes % 4096 == 0);
        CHECK_EQ(w.out, slots * kEmbd * 2);
        CHECK(w.bytes > prev);
        prev = w.bytes;
        CHECK_EQ(fixed_bytes(4, 40, rows), tables_bytes(4, 40) + 4 * w.bytes);
        CHECK_EQ(capacity(26000, 4, 40, L.stride, 1000000, rows),
                 (((int64_t)26000 << 20) - fixed_bytes(4, 40, rows)) / L.stride);
    }
    CHECK(ws(256).bytes < 42 * ((int64_t)1 << 20));   /* about 41 MB a lane at 256 rows */
    CHECK(ws(512).bytes < 82 * ((int64_t)1 << 20));
    /* 26000 MiB at 256 rows: about 8 units fewer than at 16. */
    const int64_t n16 = capacity(26000, 4, 40, L.stride, 1000000), n256 = capacity(26000, 4, 40, L.stride, 1000000, 256);
    CHECK(n256 < n16 && n16 - n256 <= 10);
    CHECK_EQ(ws(0).bytes, (int64_t)0);
    CHECK_EQ(ws(kRowsCap + 1).bytes, (int64_t)0);
    CHECK_EQ(kRowsCap, 512);
    CHECK_EQ(kCapSlots, 3072);
}

/* The deadline's scale and the shutdown line's height buckets. */
TEST(taller_deadline_and_heights) {
    for (int r = 1; r <= 16; ++r) CHECK_EQ(deadline_us_for(2000, r), (int64_t)2000);   /* today's D at 16 or fewer */
    CHECK_EQ(deadline_us_for(2000, 17), (int64_t)4000);
    CHECK_EQ(deadline_us_for(2000, 32), (int64_t)4000);
    CHECK_EQ(deadline_us_for(2000, 33), (int64_t)6000);
    CHECK_EQ(deadline_us_for(2000, 128), (int64_t)16000);
    CHECK_EQ(deadline_us_for(2000, 512), (int64_t)64000);
    CHECK_EQ(deadline_us_for(100000, 512), kDeadlineMax);   /* capped */
    CHECK_EQ(height_bucket(1), 0);
    CHECK_EQ(height_bucket(16), 0);
    CHECK_EQ(height_bucket(17), 1);
    CHECK_EQ(height_bucket(32), 1);
    CHECK_EQ(height_bucket(33), 2);
    CHECK_EQ(height_bucket(128), 3);
    CHECK_EQ(height_bucket(129), 4);
    CHECK_EQ(height_bucket(256), 4);
    CHECK_EQ(height_bucket(257), 5);
    CHECK_EQ(height_bucket(512), 5);
    CHECK_EQ(height_bucket(100000), kHeightBuckets - 1);
    for (int b = 0; b < kHeightBuckets; ++b) {
        CHECK_EQ(height_bucket(height_lo(b)), b);
        CHECK_EQ(height_bucket(height_hi(b)), b);
    }
    CHECK_EQ(height_lo(0), 1);
    CHECK_EQ(height_hi(kHeightBuckets - 1), kRowsCap);
}

TEST(quotas_follow_the_global_order_and_are_deterministic) {
    std::vector<Cand> all;
    for (int r = 0; r < 4; ++r) {
        const std::vector<Cand> p = plan(r);
        all.insert(all.end(), p.begin(), p.end());
    }
    const int64_t N = 1513;
    const std::vector<int64_t> q = quotas(all, N, 4);
    REQUIRE_EQ(q.size(), (size_t)4);
    int64_t sum = 0;
    for (int64_t x : q) sum += x;
    CHECK_EQ(sum, N);
    /* The same quotas from the globally sorted list's prefix. */
    std::vector<Cand> s = all;
    std::sort(s.begin(), s.end(), before);
    int64_t cnt[4] = {};
    for (int64_t i = 0; i < N; ++i) ++cnt[s[(size_t)i].rank];
    for (int r = 0; r < 4; ++r) CHECK_EQ(q[(size_t)r], cnt[r]);
    /* Every unit in the prefix is at least as hot as every unit past it. */
    for (int64_t i = N; i < (int64_t)s.size(); ++i) CHECK(s[(size_t)(N - 1)].share >= s[(size_t)i].share);
    /* Any input order: the same quotas and the same prefix. */
    std::mt19937 rng(12345);
    for (int t = 0; t < 5; ++t) {
        std::vector<Cand> sh = all;
        std::shuffle(sh.begin(), sh.end(), rng);
        CHECK(quotas(sh, N, 4) == q);
        std::sort(sh.begin(), sh.end(), before);
        bool same = true;
        for (int64_t i = 0; i < N; ++i) same = same && sh[(size_t)i].rank == s[(size_t)i].rank && sh[(size_t)i].unit == s[(size_t)i].unit;
        CHECK(same);
    }
    /* More room than candidates: everything. */
    const std::vector<int64_t> q2 = quotas(all, 1 << 20, 4);
    for (int r = 0; r < 4; ++r) CHECK_EQ(q2[(size_t)r], (int64_t)(40 * 96));
}

RAD_TEST_MAIN()
