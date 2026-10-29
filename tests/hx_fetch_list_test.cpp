/* hx_fetch_list_test.cpp -- a routed layer's host round trip as one ranged copy
 * (core/place/hx_fetch_list.h; FileStager with RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST).
 *
 * The list hx_fetch_ranges builds must land exactly the bytes of the separate copies the stager
 * makes without the switch (start_ep_routed's counts copy, then hx_fetch's), from sources in two
 * allocations with offsets on both sides of the base, into three pinned destinations, at every pass
 * size and both row pitches, in the stager's tables at its pass limits (16 tokens, and under
 * RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL the step: 32 and 48 here). Addresses the card's copy kernel cannot take must be left to the
 * separate copies. And a list on the card rewritten by a copy in stream order (the stager's re-arm
 * after an operand move) must not change what a ranged copy queued before the rewrite reads. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "place/ep_route_counts.h"
#include "place/hx_fetch_list.h"

#include <cstdint>
#include <cstring>
#include <vector>

using rad::HxFetch;
using rad::hx_fetch_n_ranges;
using rad::hx_fetch_ranges;

namespace {

/* The card's rows and the host's tables are sized for the tallest pass the stager takes (up to 48
 * tokens under ..._EXACT_TALL); a Host lays its sorted order out for one limit, as FileStager does. */
constexpr int64_t kEmbd = 5120, kRowsMax = rad::kEpRouteTallRows, kTopK = 6, kExperts = 96;
constexpr int64_t kSorted = kRowsMax * kTopK;      /* FileStager::hx_sorted_max() at the tall limit */
constexpr int kCap = hx_fetch_n_ranges(kRowsMax) + 1;

struct Stream {
    RadStream s = nullptr;
    Stream() { rad_stream_create(&s, 0); }
    ~Stream() { if (s) rad_stream_destroy(s); }
};

/* A rad_dev_alloc allocation, freed when it leaves scope. */
struct Mem {
    char* p;
    int kind;
    Mem(int64_t bytes, int k) : p((char*)rad_dev_alloc(bytes, k)), kind(k) {}
    ~Mem() { if (p) rad_dev_free(p, kind); }
    Mem(const Mem&) = delete;
    Mem& operator=(const Mem&) = delete;
};

void fill(char* p, int64_t n, uint32_t seed) {
    uint32_t x = seed | 1u;
    for (int64_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (char)(x >> 24);
    }
}

/* The card side: one arena holding the token rows (pitch x_ld), the sorted order and the offsets,
 * and the counts row in its own allocation or (`inside`) in the arena past the others, so that
 * every other source sits at a negative offset from the base. */
struct Card {
    Mem arena, own;
    char *counts, *x, *sorted, *off;
    int64_t x_ld;
    Card(int64_t ld, bool inside, uint32_t seed)
        : arena(4096 + kRowsMax * ld * 2 + 8192 + 4096 + 4096, RAD_MEM_DEVICE),
          own(4096, RAD_MEM_DEVICE), x_ld(ld) {
        x = arena.p + 4096 + 64;                       /* 64 past the start: not the base's alignment */
        sorted = x + kRowsMax * ld * 2 + 1024 + 4;     /* 4-byte aligned only */
        off = sorted + kSorted * 4 + 512;
        counts = inside ? off + 2048 + (16 - ((uintptr_t)(off + 2048) & 15)) % 16 : own.p;
        if (arena.p) fill(arena.p, 4096 + kRowsMax * ld * 2 + 8192 + 4096 + 4096, seed);
        if (own.p) fill(own.p, 4096, seed * 7 + 1);
    }
    bool ok() const { return arena.p && own.p; }
};

/* The host side, laid out as FileStager's at pass limit `lim`: counts_host_ (512 entries), hx_x_
 * (lim packed rows) and hx_idx_ (sorted order, then the offsets at hx_sorted_max() = lim * 6). */
struct Host {
    Mem counts, x, idx;
    int64_t lim;
    explicit Host(int64_t limit = 16)
        : counts(512 * 4, RAD_MEM_HOST_PINNED), x(kRowsMax * kEmbd * 2, RAD_MEM_HOST_PINNED),
          idx((kSorted + 128) * 4, RAD_MEM_HOST_PINNED), lim(limit) {}
    bool ok() const { return counts.p && x.p && idx.p; }
    void clear() {
        std::memset(counts.p, 0xEE, 512 * 4);
        std::memset(x.p, 0xEE, kRowsMax * kEmbd * 2);
        std::memset(idx.p, 0xEE, (kSorted + 128) * 4);
    }
    bool same(const Host& o) const {
        return !std::memcmp(counts.p, o.counts.p, 512 * 4) &&
               !std::memcmp(x.p, o.x.p, kRowsMax * kEmbd * 2) &&
               !std::memcmp(idx.p, o.idx.p, (kSorted + 128) * 4);
    }
};

HxFetch fetch(const Card& c, const Host& h, int64_t rows) {
    HxFetch f;
    f.counts = c.counts;
    f.counts_host = (int32_t*)h.counts.p;
    f.n_counts = kExperts;
    f.x = c.x;
    f.x_ld = c.x_ld;
    f.x_host = (uint16_t*)h.x.p;
    f.embd = kEmbd;
    f.rows = rows;
    f.sorted = c.sorted;
    f.sorted_host = (int32_t*)h.idx.p;
    f.n_sorted = rows * kTopK;
    f.off = c.off;
    f.off_host = (int32_t*)h.idx.p + h.lim * kTopK;
    f.n_off = kExperts + 1;
    return f;
}

/* What the stager queues without the switch: the counts row, then hx_fetch's copies. */
int separate(const HxFetch& f, RadStream s) {
    RAD_TRY(rad_memcpy_async(f.counts_host, f.counts, f.n_counts * 4, s));
    if (f.x_ld == f.embd) {
        RAD_TRY(rad_memcpy_async(f.x_host, f.x, f.rows * f.embd * 2, s));
    } else {
        for (int64_t r = 0; r < f.rows; ++r)
            RAD_TRY(rad_memcpy_async(f.x_host + r * f.embd, (const char*)f.x + r * f.x_ld * 2,
                                     f.embd * 2, s));
    }
    RAD_TRY(rad_memcpy_async(f.sorted_host, f.sorted, f.n_sorted * 4, s));
    RAD_TRY(rad_memcpy_async(f.off_host, f.off, f.n_off * 4, s));
    return RAD_OK;
}

/* The stager's arming: the list staged in pinned memory and copied onto the card in stream order. */
int arm(RadCopyRange* dev, RadCopyRange* pin, const RadCopyRange* r, int n, RadStream s) {
    std::memcpy(pin, r, (size_t)n * sizeof(RadCopyRange));
    return rad_memcpy_async(dev, pin, (int64_t)n * (int64_t)sizeof(RadCopyRange), s);
}

}  // namespace

TEST(a_fetch_list_lands_what_the_separate_copies_land) {
    Stream st;
    REQUIRE(st.s);
    Mem dev(kCap * (int64_t)sizeof(RadCopyRange), RAD_MEM_DEVICE);
    Mem pin(kCap * (int64_t)sizeof(RadCopyRange), RAD_MEM_HOST_PINNED);
    REQUIRE(dev.p && pin.p);
    uint32_t seed = 11;
    for (int64_t lim : { (int64_t)16, (int64_t)32, kRowsMax })
    for (bool inside : { false, true })
        for (int64_t ld : { kEmbd, kEmbd + 64 })
            for (int64_t rows : { 1, 2, 7, 16, 17, 33, 48 }) {
                if (rows > lim) continue;
                Host a(lim), b(lim);
                REQUIRE(a.ok() && b.ok());
                Card c(ld, inside, ++seed);
                REQUIRE(c.ok());
                a.clear();
                b.clear();
                const HxFetch fa = fetch(c, a, rows), fb = fetch(c, b, rows);
                RadCopyRange r[kCap];
                const int n = hx_fetch_ranges(fa, r, kCap);
                CHECK_EQ(n, hx_fetch_n_ranges(rows));
                if (n <= 0) continue;
                /* In the arena case every source but the counts lies below the base. */
                if (inside) CHECK(r[1].src < 0 && r[n - 1].src < 0);
                CHECK_OK(arm((RadCopyRange*)dev.p, (RadCopyRange*)pin.p, r, n, st.s));
                CHECK_OK(rad_memcpy_ranges_async(fa.counts_host, fa.counts, 0,
                                                 (const RadCopyRange*)dev.p, n, st.s));
                CHECK_OK(separate(fb, st.s));
                CHECK_OK(rad_stream_sync(st.s));
                CHECK(a.same(b));
                /* ...and both moved the sources' bytes, not nothing. */
                CHECK_EQ(std::memcmp(a.counts.p, c.counts, kExperts * 4), 0);
                CHECK_EQ(std::memcmp(a.x.p + (rows - 1) * kEmbd * 2, c.x + (rows - 1) * ld * 2,
                                     kEmbd * 2), 0);
                CHECK_EQ(std::memcmp(a.idx.p, c.sorted, rows * kTopK * 4), 0);
                CHECK_EQ(std::memcmp(a.idx.p + lim * kTopK * 4, c.off, (kExperts + 1) * 4), 0);
                /* The bytes past the pass's rows are not touched. */
                if (rows < kRowsMax) CHECK_EQ((unsigned char)a.x.p[rows * kEmbd * 2], 0xEEu);
            }
}

TEST(a_fetch_list_leaves_what_the_copy_kernel_cannot_take_to_separate_copies) {
    Host h;
    REQUIRE(h.ok());
    Card c(kEmbd, false, 5), odd(kEmbd + 1, false, 6);
    REQUIRE(c.ok() && odd.ok());
    RadCopyRange r[kCap];
    HxFetch f = fetch(c, h, 4);
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 7);
    /* The kernel's bases are 16-byte aligned. */
    f.counts = c.counts + 4;
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 0);
    f = fetch(c, h, 4);
    f.counts_host += 1;
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 0);
    /* An odd pitch puts every second row 2 bytes off a word: fine for one row, not for two. */
    f = fetch(odd, h, 1);
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 4);
    f = fetch(odd, h, 2);
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 0);
    /* Room for every range, rows no shorter than a row, and something to copy. */
    f = fetch(c, h, 16);
    CHECK_EQ(hx_fetch_ranges(f, r, hx_fetch_n_ranges(16) - 1), 0);
    CHECK_EQ(hx_fetch_ranges(f, r, hx_fetch_n_ranges(16)), hx_fetch_n_ranges(16));
    /* The tall limit's pass: 51 ranges, the stager's slot of 52 whole pairs. */
    f = fetch(c, h, 48);
    CHECK_EQ(hx_fetch_ranges(f, r, hx_fetch_n_ranges(48) - 1), 0);
    CHECK_EQ(hx_fetch_ranges(f, r, hx_fetch_n_ranges(48)), 51);
    f.x_ld = kEmbd - 2;
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 0);
    f = fetch(c, h, 0);
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 0);
    f = fetch(c, h, 2);
    f.sorted = nullptr;
    CHECK_EQ(hx_fetch_ranges(f, r, kCap), 0);
}

/* The stager re-arms a layer's list (after its gate_up's operands moved) by a copy onto the card in
 * stream order, from a staging slot it rewrites only once an event says the slot's last copy has
 * left. A ranged copy queued before the re-arm must still read the old list. Here the old list
 * fetches three rows from one card and the new one a single row from another: rows 1 and 2 must
 * hold the first card's bytes, row 0 and the counts the second's. */
TEST(a_fetch_list_rewritten_in_stream_order_leaves_earlier_copies_their_list) {
    Stream st;
    REQUIRE(st.s);
    RadEvent ev = nullptr;
    REQUIRE(rad_event_create(&ev) >= 0);
    Host h;
    REQUIRE(h.ok());
    Card one(kEmbd, false, 21), two(kEmbd + 64, true, 22);
    REQUIRE(one.ok() && two.ok());
    Mem dev(kCap * (int64_t)sizeof(RadCopyRange), RAD_MEM_DEVICE);
    Mem pin(kCap * (int64_t)sizeof(RadCopyRange), RAD_MEM_HOST_PINNED);
    REQUIRE(dev.p && pin.p);
    h.clear();
    RadCopyRange r1[kCap], r2[kCap];
    const HxFetch f1 = fetch(one, h, 3), f2 = fetch(two, h, 1);
    const int n1 = hx_fetch_ranges(f1, r1, kCap), n2 = hx_fetch_ranges(f2, r2, kCap);
    REQUIRE(n1 == 6 && n2 == 4);
    CHECK_OK(arm((RadCopyRange*)dev.p, (RadCopyRange*)pin.p, r1, n1, st.s));
    CHECK_OK(rad_event_record(ev, st.s));
    CHECK_OK(rad_memcpy_ranges_async(f1.counts_host, f1.counts, 0, (const RadCopyRange*)dev.p, n1,
                                     st.s));
    CHECK_OK(rad_event_sync(ev));   /* the staging slot's copy has left: it may be rewritten */
    CHECK_OK(arm((RadCopyRange*)dev.p, (RadCopyRange*)pin.p, r2, n2, st.s));
    CHECK_OK(rad_memcpy_ranges_async(f2.counts_host, f2.counts, 0, (const RadCopyRange*)dev.p, n2,
                                     st.s));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ(std::memcmp(h.x.p, two.x, kEmbd * 2), 0);
    CHECK_EQ(std::memcmp(h.x.p + kEmbd * 2, one.x + kEmbd * 2, 2 * kEmbd * 2), 0);
    CHECK_EQ(std::memcmp(h.counts.p, two.counts, kExperts * 4), 0);
    CHECK_EQ(std::memcmp(h.idx.p, two.sorted, kTopK * 4), 0);
    CHECK_EQ(std::memcmp(h.idx.p + kTopK * 4, one.sorted + kTopK * 4, 2 * kTopK * 4), 0);
    CHECK_EQ(std::memcmp(h.idx.p + h.lim * kTopK * 4, two.off, (kExperts + 1) * 4), 0);
    rad_event_destroy(ev);
}

RAD_TEST_MAIN()
